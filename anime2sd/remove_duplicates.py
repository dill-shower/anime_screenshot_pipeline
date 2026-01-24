import safetensors
import os
os.environ["PYTORCH_INDUCTOR_CACHE_DIR"] = "/home/user/cache"
import logging
from tqdm import tqdm
from typing import List, Tuple, Set, Optional
import pillow_jxl
from PIL import Image
from pathlib import Path
import subprocess
import time
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from .basics import get_related_paths, get_images_recursively
import warnings
import torch.backends.cudnn as cudnn
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning)

# FAISS support
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform: callable):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image_path, image

    @classmethod
    def from_directory(cls, dataset_dir: str, transform: callable):
        image_paths = get_images_recursively(dataset_dir)
        return cls(image_paths, transform)

    @classmethod
    def from_subdirectories(cls, dataset_dir: str, transform: callable, portion: Optional[str] = "first"):
        def get_image_number(filename):
            return int(os.path.splitext(filename)[0].split("_")[-1])

        image_paths = []
        for subdir in os.listdir(dataset_dir):
            subdir_path = os.path.join(dataset_dir, subdir)
            if os.path.isdir(subdir_path):
                image_files = get_images_recursively(subdir_path)
                if not image_files:
                    continue
                image_numbers = [get_image_number(f) for f in image_files]
                sorted_files = [x for _, x in sorted(zip(image_numbers, image_files))]

                max_number = max(image_numbers)
                threshold = max_number // 3

                if portion == "first":
                    selected_files = [f for f in sorted_files if get_image_number(f) <= threshold]
                elif portion == "last":
                    selected_files = [f for f in sorted_files if get_image_number(f) > 2 * threshold]
                else:
                    raise ValueError("portion must be either 'first' or 'last'")

                image_paths.extend(selected_files)

        return cls(image_paths, transform)


def _is_distributed_initialized() -> bool:
    """Проверяет, инициализировано ли распределённое окружение"""
    return dist.is_available() and dist.is_initialized()


def _is_launched_with_torchrun() -> bool:
    """Проверяет, запущен ли скрипт через torchrun/torch.distributed.launch"""
    return all(key in os.environ for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])


class DuplicateRemover:
    def __init__(
        self, 
        model_name: str, 
        device: Optional[str] = None, 
        threshold: float = 0.96,
        max_compare_size: int = 200000, 
        dataloader_batch_size: int = 64, 
        dataloader_num_workers: int = 124, 
        pin_memory: bool = True,
        use_faiss: bool = True,
        faiss_exact_threshold: int = 40000,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger()
        
        # FAISS configuration
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_exact_threshold = faiss_exact_threshold
        
        if use_faiss and not FAISS_AVAILABLE:
            self.logger.warning("⚠️ FAISS requested but not available!")
            self.logger.warning("   Install with: pip install faiss-gpu")
            self.logger.warning("   Falling back to sklearn cosine_similarity")
            self.use_faiss = False
        
        # Multi-GPU setup
        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Проверяем, запущен ли скрипт в распределённом режиме
        if _is_launched_with_torchrun():
            # Запущено через torchrun — инициализируем DDP
            if not _is_distributed_initialized():
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
                
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=self.world_size,
                    rank=int(os.environ.get("RANK", 0))
                )
                self.logger.info(f"✅ Initialized DDP: rank={self.local_rank}, world_size={self.world_size}")
            else:
                self.local_rank = dist.get_rank()
            
            self.use_ddp = True
            self.use_dp = False
        elif self.world_size > 1:
            # Несколько GPU, но обычный запуск — используем DataParallel
            self.local_rank = 0
            self.use_ddp = False
            self.use_dp = True
            self.logger.info(f"🔧 Using DataParallel with {self.world_size} GPUs")
        else:
            # Одна GPU или CPU
            self.local_rank = 0
            self.use_ddp = False
            self.use_dp = False
        
        if torch.cuda.is_available():
            self.device = f"cuda:{self.local_rank}" if self.use_ddp else "cuda:0"
        else:
            self.device = "cpu"
            self.use_ddp = False
            self.use_dp = False
            self.use_faiss = False

        # GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
            torch.use_deterministic_algorithms(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        self.model = None
        self.threshold = threshold
        self.max_compare_size = max_compare_size
        
        # Batch size calculation
        if self.use_ddp:
            self.effective_batch_size = dataloader_batch_size
            self.dataloader_batch_size = dataloader_batch_size // self.world_size
        elif self.use_dp:
            # DataParallel автоматически распределяет батч
            self.effective_batch_size = dataloader_batch_size
            self.dataloader_batch_size = dataloader_batch_size
        else:
            self.effective_batch_size = dataloader_batch_size
            self.dataloader_batch_size = dataloader_batch_size
            
        self.dataloader_num_workers = min(
            os.cpu_count() // max(self.world_size, 1), 
            dataloader_num_workers
        )
        self.pin_memory = pin_memory
        
        self.model_name = model_name
        self._initialize_model()
        
        # Log configuration
        if self.local_rank == 0:
            self.logger.info(f"{'='*70}")
            self.logger.info(f"🔧 DuplicateRemover Configuration:")
            self.logger.info(f"   Similarity backend: {'FAISS' if self.use_faiss else 'sklearn'}")
            if self.use_faiss:
                self.logger.info(f"   FAISS exact search up to: {self.faiss_exact_threshold:,} images")
                self.logger.info(f"   Above {self.faiss_exact_threshold:,}: IVF approximate search")
            self.logger.info(f"   Transitivity: ENABLED (groups similar images)")
            self.logger.info(f"   Threshold: {self.threshold}")
            self.logger.info(f"   GPUs: {self.world_size}")
            
            if self.use_ddp:
                self.logger.info(f"   Mode: DistributedDataParallel (DDP)")
                self.logger.info(f"   Batch size: {self.effective_batch_size} ({self.dataloader_batch_size}/GPU)")
            elif self.use_dp:
                self.logger.info(f"   Mode: DataParallel (DP)")
                self.logger.info(f"   Batch size: {self.dataloader_batch_size}")
            else:
                self.logger.info(f"   Mode: Single GPU/CPU")
                self.logger.info(f"   Batch size: {self.dataloader_batch_size}")
            self.logger.info(f"{'='*70}\n")

    def _initialize_model(self):
        """Инициализация модели"""
        if self.local_rank == 0:
            mode = "DDP" if self.use_ddp else ("DataParallel" if self.use_dp else "Single")
            self.logger.info(f"🚀 Initializing model ({mode} mode, {self.world_size} GPU(s))")
        
        self.model = self.load_model_on_gpus(self.model_name)
        
        # Получаем pretrained_cfg в зависимости от типа обёртки
        if self.use_ddp or self.use_dp:
            base_model = self.model.module
        else:
            base_model = self.model
        
        # Для torch.compile модель может быть обёрнута
        if hasattr(base_model, 'pretrained_cfg'):
            self.data_cfg = timm.data.resolve_data_config(base_model.pretrained_cfg)
        elif hasattr(base_model, '_orig_mod') and hasattr(base_model._orig_mod, 'pretrained_cfg'):
            self.data_cfg = timm.data.resolve_data_config(base_model._orig_mod.pretrained_cfg)
        else:
            # Fallback: загружаем конфиг отдельно
            temp_model = timm.create_model(self.model_name, pretrained=False)
            self.data_cfg = timm.data.resolve_data_config(temp_model.pretrained_cfg)
            del temp_model
            
        self.transform = timm.data.create_transform(**self.data_cfg)

    def load_model_on_gpus(self, model_name):
        """Загрузка модели с оптимизациями"""
        if self.local_rank == 0:
            self.logger.info(f"Loading model: {model_name}")
        
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.to(self.device)
            
            # ================================================================
            # torch.compile ТОЛЬКО для single GPU или DDP
            # DataParallel несовместим с torch.compile!
            # ================================================================
            if not self.use_dp:
                if self.local_rank == 0:
                    self.logger.info("🔥 Compiling model with torch.compile(mode='max-autotune')")
                
                try:
                    model = torch.compile(model, mode="max-autotune", fullgraph=False)
                    if self.local_rank == 0:
                        self.logger.info("✅ torch.compile enabled")
                except Exception as e:
                    if self.local_rank == 0:
                        self.logger.warning(f"⚠️ torch.compile failed: {e}")
            else:
                if self.local_rank == 0:
                    self.logger.info("⚠️ torch.compile skipped (incompatible with DataParallel)")
            
            # Выбираем способ параллелизации
            if self.use_ddp:
                model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            elif self.use_dp:
                model = DataParallel(model)
                if self.local_rank == 0:
                    self.logger.info(f"✅ DataParallel enabled on {self.world_size} GPUs")
            
            # Warmup
            with torch.no_grad():
                test_input = torch.randn(1, 3, 448, 448).to(self.device)
                _ = model(test_input)
            
            if self.local_rank == 0:
                self.logger.info("✅ Model verification successful")
                mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2
                self.logger.info(f"   GPU memory: {mem_allocated:.0f}MB allocated, {mem_reserved:.0f}MB reserved")
        
        return model
    
    def compute_embeddings(self, dataset: ImageDataset) -> np.ndarray:
        """Вычисление эмбеддингов"""
        if self.use_ddp:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.dataloader_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            shuffle=False,
            sampler=sampler,
            persistent_workers=True if self.dataloader_num_workers > 0 else False,
        )

        embeddings = []
        with torch.no_grad():
            for _, images in tqdm(
                dataloader, 
                desc="Computing embeddings", 
                leave=False,
                disable=self.use_ddp and self.local_rank != 0
            ):
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                embeddings.append(features.cpu().numpy())

        local_embeddings = np.vstack(embeddings)
        
        if self.use_ddp:
            # Gather embeddings from all GPUs
            local_size = torch.tensor([local_embeddings.shape[0]], device=self.device)
            sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
            dist.all_gather(sizes, local_size)
            
            max_size = max([s.item() for s in sizes])
            padded_embeddings = np.zeros((max_size, local_embeddings.shape[1]), dtype=local_embeddings.dtype)
            padded_embeddings[:local_embeddings.shape[0]] = local_embeddings
            
            tensor_embeddings = torch.from_numpy(padded_embeddings).to(self.device)
            gathered = [torch.zeros_like(tensor_embeddings) for _ in range(self.world_size)]
            dist.all_gather(gathered, tensor_embeddings)
            
            all_embeddings = []
            for i, size in enumerate(sizes):
                all_embeddings.append(gathered[i][:size.item()].cpu().numpy())
            
            return np.vstack(all_embeddings)
        
        return local_embeddings

    def get_file_sizes(self, indices: np.ndarray) -> dict:
        """Получить размеры файлов параллельно"""
        with ThreadPoolExecutor(max_workers=min(120, len(indices))) as executor:
            paths = [self.dataset.image_paths[idx] for idx in indices]
            sizes = list(executor.map(self._get_file_size, paths))
            return dict(zip(indices, sizes))
    
    @staticmethod
    def _get_file_size(file_path: str) -> int:
        try:
            return os.path.getsize(file_path)
        except:
            return 0

    def find_duplicates_faiss(self, embeddings: np.ndarray) -> Set[int]:
        """
        Поиск дубликатов с FAISS — С транзитивностью (как в оригинале).
        До faiss_exact_threshold — точный поиск, выше — IVF.
        """
        n, d = embeddings.shape
        
        if self.local_rank == 0:
            self.logger.info(f"🔍 Finding duplicates using FAISS (n={n:,}, d={d})")
        
        # Получаем размеры файлов
        if self.local_rank == 0:
            self.logger.info("   Getting file sizes...")
        
        all_indices = np.arange(n)
        file_sizes = self.get_file_sizes(all_indices)
        file_sizes_array = np.array([file_sizes[i] for i in range(n)])
        
        # Нормализуем для косинусного сходства
        if self.local_rank == 0:
            self.logger.info("   Normalizing embeddings...")
        
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Выбираем тип индекса
        use_exact = n <= self.faiss_exact_threshold
        
        if use_exact:
            if self.local_rank == 0:
                self.logger.info(f"   Using EXACT search (IndexFlatIP)")
                self.logger.info(f"   Accuracy: 100% (identical to full matrix)")
            index = faiss.IndexFlatIP(d)
        else:
            nlist = min(int(4 * np.sqrt(n)), 4096)
            nprobe = max(nlist // 2, 128)
            
            if self.local_rank == 0:
                self.logger.info(f"   Using IVF approximate search")
                self.logger.info(f"   nlist={nlist}, nprobe={nprobe}")
                self.logger.info(f"   Expected accuracy: >99.5%")
            
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = nprobe
        
        # GPU acceleration
        if torch.cuda.is_available():
            try:
                if self.local_rank == 0:
                    self.logger.info("   Moving FAISS index to GPU...")
                
                res = faiss.StandardGpuResources()
                # Увеличиваем временную память для больших индексов
                res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB
                
                index = faiss.index_cpu_to_gpu(res, 0, index)
                
                if self.local_rank == 0:
                    self.logger.info("   ✅ FAISS using GPU acceleration")
            except Exception as e:
                if self.local_rank == 0:
                    self.logger.warning(f"   ⚠️ GPU FAISS failed: {e}, using CPU")
        
        # Обучаем если нужно
        if hasattr(index, 'is_trained') and not index.is_trained:
            if self.local_rank == 0:
                self.logger.info("   Training index...")
            
            train_size = min(n, 100000)
            if train_size < n:
                train_indices = np.random.choice(n, train_size, replace=False)
                index.train(embeddings[train_indices])
            else:
                index.train(embeddings)
        
        # Добавляем векторы
        if self.local_rank == 0:
            self.logger.info("   Adding vectors to index...")
        index.add(embeddings)
        
        # Ищем k ближайших соседей (больше для транзитивности)
        k = min(100, n)
        
        if self.local_rank == 0:
            self.logger.info(f"   Searching for {k} nearest neighbors per image...")
        
        search_start = time.time()
        similarities, neighbor_indices = index.search(embeddings, k)
        search_time = time.time() - search_start
        
        if self.local_rank == 0:
            self.logger.info(f"   Search completed in {search_time:.2f}s")
            self.logger.info(f"   Throughput: {n*k/search_time:,.0f} comparisons/sec")
        
        # ========================================================================
        # ОРИГИНАЛЬНАЯ ЛОГИКА С ТРАНЗИТИВНОСТЬЮ
        # ========================================================================
        
        if self.local_rank == 0:
            self.logger.info("   Processing with transitivity (grouping similar images)...")
        
        samples_to_remove = set()
        samples_to_keep = set()
        groups_found = 0
        
        for i in tqdm(
            range(n), 
            desc="   Finding duplicate groups", 
            leave=False,
            disable=self.local_rank != 0
        ):
            if i in samples_to_remove or i in samples_to_keep:
                continue
            
            # Находим все похожие (включая себя)
            similar_mask = similarities[i] > self.threshold
            dup_local_idxs = np.where(similar_mask)[0]
            
            if len(dup_local_idxs) > 1:  # Есть дубликаты кроме себя
                # Собираем ВСЕ похожие индексы (транзитивность!)
                similar_images = [int(neighbor_indices[i, idx]) for idx in dup_local_idxs]
                similar_images = [x for x in similar_images if x != -1]
                
                if len(similar_images) > 1:
                    groups_found += 1
                    
                    # Оставляем самый БОЛЬШОЙ файл из группы
                    largest_image = max(similar_images, key=lambda x: file_sizes_array[x])
                    
                    samples_to_keep.add(largest_image)
                    samples_to_remove.update(set(similar_images) - {largest_image})
            else:
                # Нет дубликатов
                samples_to_keep.add(i)
        
        if self.local_rank == 0:
            self.logger.info(f"   Duplicate groups found: {groups_found:,}")
            self.logger.info(f"   Files to remove: {len(samples_to_remove):,}")
            self.logger.info(f"   Files to keep: {len(samples_to_keep):,}")
        
        return samples_to_remove

    def find_duplicates_sklearn(self, embeddings: np.ndarray) -> Set[int]:
        """
        Fallback метод без FAISS (батчами, с транзитивностью)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        n = len(embeddings)
        
        if self.local_rank == 0:
            self.logger.info(f"🔍 Finding duplicates using sklearn (n={n:,})")
            if n > 50000:
                mem_needed_gb = (n * n * 4) / (1024**3)
                self.logger.warning(f"   ⚠️ Large dataset! Memory needed: ~{mem_needed_gb:.1f} GB")
                self.logger.warning(f"   Consider installing faiss-gpu for better performance")
        
        batch_size = 1280
        remaining_indices = np.arange(n)
        all_to_remove = set()
        batch_num = 0
        
        while len(remaining_indices) > 0:
            batch_num += 1
            current_batch_size = min(batch_size, len(remaining_indices))
            
            if current_batch_size < len(remaining_indices):
                batch_indices = np.random.choice(remaining_indices, current_batch_size, replace=False)
            else:
                batch_indices = remaining_indices
    
            if self.local_rank == 0:
                self.logger.info(
                    f"Batch {batch_num}: processing {current_batch_size} images "
                    f"({len(remaining_indices)} remaining)"
                )
            
            batch_embeddings = embeddings[batch_indices]
            similarity_matrix = cosine_similarity(batch_embeddings)
            similarity_matrix = similarity_matrix - np.identity(len(similarity_matrix))
    
            samples_to_remove = set()
            samples_to_keep = set()
            
            file_sizes = self.get_file_sizes(batch_indices)
    
            for idx in range(len(batch_embeddings)):
                sample_id = batch_indices[idx]
                if sample_id not in samples_to_remove and sample_id not in samples_to_keep:
                    dup_idxs = np.where(similarity_matrix[idx] > self.threshold)[0]
                    
                    if len(dup_idxs) > 0:
                        # ТРАНЗИТИВНОСТЬ: группируем все похожие
                        similar_images = [sample_id] + [batch_indices[dup] for dup in dup_idxs]
                        largest_image = max(similar_images, key=lambda x: file_sizes[x])
                        samples_to_keep.add(largest_image)
                        samples_to_remove.update(set(similar_images) - {largest_image})
                    else:
                        samples_to_keep.add(sample_id)
    
            all_to_remove.update(samples_to_remove)
            remaining_indices = np.array([
                idx for idx in remaining_indices 
                if idx not in samples_to_remove and idx not in samples_to_keep
            ])
            
            if self.local_rank == 0:
                self.logger.info(f"   Found {len(samples_to_remove)} duplicates in this batch")
        
        if self.local_rank == 0:
            self.logger.info(f"   Total files to remove: {len(all_to_remove):,}")
        
        return all_to_remove

    def remove_similar(self, dataset: ImageDataset):
        """Основной метод удаления дубликатов"""
        start_time = time.time()
        
        if self.local_rank == 0:
            self.logger.info(f"Starting duplicate removal for {len(dataset):,} images...")
        
        # Вычисляем эмбеддинги
        embeddings = self.compute_embeddings(dataset)
        embedding_time = time.time() - start_time
        
        if self.local_rank == 0:
            self.logger.info(f"✅ Embedding computation: {embedding_time:.2f}s")
            self.logger.info(f"   Throughput: {len(dataset)/embedding_time:.1f} images/sec\n")
    
        # Только на rank 0 (или single GPU / DataParallel)
        if self.use_ddp and self.local_rank != 0:
            dist.barrier()
            return
        
        # Сохраняем dataset для get_file_sizes
        self.dataset = dataset
        
        # Поиск дубликатов
        search_start = time.time()
        
        if self.use_faiss:
            to_remove = self.find_duplicates_faiss(embeddings)
        else:
            to_remove = self.find_duplicates_sklearn(embeddings)
        
        search_time = time.time() - search_start
        
        self.logger.info(f"\n✅ Duplicate search: {search_time:.2f}s\n")
        
        # Удаляем файлы
        if to_remove:
            files_to_remove = [dataset.image_paths[idx] for idx in to_remove]
            
            # Подсчитываем размер
            total_size = 0
            for path in files_to_remove[:1000]:  # Sample для оценки
                try:
                    total_size += os.path.getsize(path)
                except:
                    pass
            
            if len(files_to_remove) > 0:
                estimated_total_gb = (total_size / min(1000, len(files_to_remove)) * len(files_to_remove)) / (1024**3)
            else:
                estimated_total_gb = 0
            
            self.logger.info(f"🗑️  Removing {len(files_to_remove):,} duplicate files...")
            self.logger.info(f"   Estimated space to free: ~{estimated_total_gb:.2f} GB")
            
            removal_start = time.time()
            batch_size = 1000
            removed_count = 0
            
            for i in range(0, len(files_to_remove), batch_size):
                batch = files_to_remove[i:i + batch_size]
                try:
                    subprocess.run(["rm"] + batch, check=True, stderr=subprocess.DEVNULL)
                    removed_count += len(batch)
                    
                    if (i // batch_size + 1) % 10 == 0:
                        progress = removed_count / len(files_to_remove) * 100
                        self.logger.info(f"   Progress: {removed_count:,}/{len(files_to_remove):,} ({progress:.1f}%)")
                except Exception as e:
                    self.logger.error(f"   Error removing batch: {e}")
            
            removal_time = time.time() - removal_start
            self.logger.info(f"✅ Removed {removed_count:,} files in {removal_time:.2f}s")
            self.logger.info(f"   Throughput: {removed_count/removal_time:,.0f} files/sec")
        else:
            self.logger.info("✅ No duplicates found")
        
        total_time = time.time() - start_time
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📊 SUMMARY:")
        self.logger.info(f"   Total images: {len(dataset):,}")
        self.logger.info(f"   Duplicates removed: {len(to_remove):,}")
        self.logger.info(f"   Unique images kept: {len(dataset) - len(to_remove):,}")
        self.logger.info(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        self.logger.info(f"{'='*70}\n")
        
        if self.use_ddp:
            dist.barrier()

    def remove_similar_from_dir(self, dirpath: str, portion: Optional[str] = None):
        """Удаление дубликатов из директории"""
        if portion:
            rtype = "OP" if portion == "first" else "ED"
            if self.local_rank == 0:
                self.logger.info(f"{'='*70}")
                self.logger.info(f"Processing {rtype} portion: {dirpath}")
                self.logger.info(f"{'='*70}\n")
            
            dataset = ImageDataset.from_subdirectories(
                dataset_dir=dirpath, transform=self.transform, portion=portion
            )
        else:
            if self.local_rank == 0:
                self.logger.info(f"{'='*70}")
                self.logger.info(f"Processing directory: {dirpath}")
                self.logger.info(f"{'='*70}\n")
            
            dataset = ImageDataset.from_directory(dirpath, self.transform)

        if len(dataset) == 0:
            if self.local_rank == 0:
                self.logger.warning("No images found!")
            return

        self.dataset = dataset
        self.remove_similar(dataset)

    @staticmethod
    def get_file_size(file_path):
        """Статический метод для обратной совместимости"""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    def __del__(self):
        """Cleanup при удалении объекта"""
        if self.use_ddp and _is_distributed_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    remover = DuplicateRemover(
        model_name="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        threshold=0.91,
        dataloader_batch_size=64,
        dataloader_num_workers=32,
        use_faiss=True,
        faiss_exact_threshold=40000,
        logger=logger
    )
    
    remover.remove_similar_from_dir("./dataset")

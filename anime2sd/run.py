import pillow_jxl
import os
os.environ["PYTORCH_INDUCTOR_CACHE_DIR"] = "/home/user/cache"
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
import timm
import numpy as np
import argparse
import pillow_jxl
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Install with: pip install faiss-gpu (or faiss-cpu)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_distributed_initialized() -> bool:
    """Проверяет, инициализировано ли распределённое окружение"""
    return dist.is_available() and dist.is_initialized()


def _is_launched_with_torchrun() -> bool:
    """Проверяет, запущен ли скрипт через torchrun/torch.distributed.launch"""
    return all(key in os.environ for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])


def setup_optimizations():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info("Enabling TF32 precision...")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            logger.info("GPU does not support TF32, using standard precision...")
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("✅ PyTorch SDPA (Flash Attention / Memory Efficient) enabled")
        torch.use_deterministic_algorithms(False)
        return True
    else:
        logger.info("CUDA not available, using CPU...")
        return False


class ImageDataset(Dataset):
    def __init__(self, directory, transform):
        self.image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.jxl', '.webp', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
            image.load()
            image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return image_path, image, False
        except Exception as e:
            logger.warning(f"⚠️  Broken image: {image_path} | Error: {type(e).__name__}: {e}")
            return image_path, None, True


def safe_collate_fn(batch):
    valid_items = [(path, img) for path, img, is_broken in batch if not is_broken]
    broken_paths = [path for path, img, is_broken in batch if is_broken]
    
    if len(valid_items) == 0:
        return [], None, broken_paths
    
    paths, images = zip(*valid_items)
    return list(paths), torch.stack(images), broken_paths


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_and_compile_model(model_name, device, use_ddp=False, use_dp=False, rank=0):
    """
    Загружает и компилирует модель.
    
    Args:
        model_name: Название модели из timm
        device: Устройство (cuda:X или cpu)
        use_ddp: Использовать DistributedDataParallel
        use_dp: Использовать DataParallel
        rank: Rank процесса (для логов)
    """
    if rank == 0:
        mode = "DDP" if use_ddp else ("DataParallel" if use_dp else "Single GPU")
        logger.info(f"Loading model: {model_name} ({mode} mode)")
    
    base_model = timm.create_model(model_name, pretrained=True)
    base_model.eval()
    
    if torch.cuda.is_available():
        base_model = base_model.to(device)
        
        # ================================================================
        # torch.compile ТОЛЬКО для single GPU или DDP
        # DataParallel несовместим с torch.compile!
        # ================================================================
        if not use_dp:
            if rank == 0:
                logger.info("🔥 Compiling model with torch.compile(mode='max-autotune')")
            
            try:
                compiled_model = torch.compile(
                    base_model,
                    mode="max-autotune",
                    fullgraph=False
                )
                
                with torch.no_grad():
                    test_input = torch.randn(1, 3, 448, 448).to(device)
                    _ = compiled_model(test_input)
                
                if rank == 0:
                    logger.info("✅ Model compiled successfully with max-autotune")
                model = compiled_model
            except Exception as e:
                if rank == 0:
                    logger.warning(f"⚠️  Compilation failed: {e}")
                    logger.info("Using uncompiled model")
                model = base_model
        else:
            if rank == 0:
                logger.info("⚠️ torch.compile skipped (incompatible with DataParallel)")
            model = base_model
        
        # Применяем параллелизацию
        if use_ddp:
            model = DDP(model, device_ids=[rank], output_device=rank)
            if rank == 0:
                logger.info(f"✅ DDP initialized on {dist.get_world_size()} GPUs")
        elif use_dp:
            model = DataParallel(model)
            if rank == 0:
                world_size = torch.cuda.device_count()
                logger.info(f"✅ DataParallel enabled on {world_size} GPUs")
    else:
        model = base_model
    
    model.eval()
    return model


def compute_embeddings_ddp(model, dataloader, device, rank, world_size, use_ddp):
    embeddings = []
    paths = []
    all_broken_files = []
    skipped_batches = 0
    
    with torch.no_grad():
        for batch_paths, images, broken_paths in tqdm(dataloader, desc="Computing embeddings", 
                                                       disable=use_ddp and rank != 0):
            all_broken_files.extend(broken_paths)
            
            if images is None or len(batch_paths) == 0:
                skipped_batches += 1
                continue
            
            images = images.to(device, non_blocking=True)
            features = model(images)
            embeddings.append(features.cpu().numpy())
            paths.extend(batch_paths)
    
    if skipped_batches > 0 and (rank == 0 or not use_ddp):
        logger.warning(f"⚠️  Skipped {skipped_batches} fully broken batches")
    
    local_embeddings = np.vstack(embeddings) if embeddings else np.array([]).reshape(0, 0)
    
    if use_ddp:
        local_size = torch.tensor([local_embeddings.shape[0]], device=device)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        
        if local_embeddings.shape[0] > 0:
            max_size = max([s.item() for s in sizes])
            embed_dim = local_embeddings.shape[1]
            padded_embeddings = np.zeros((max_size, embed_dim), dtype=local_embeddings.dtype)
            padded_embeddings[:local_embeddings.shape[0]] = local_embeddings
        else:
            max_size = max([s.item() for s in sizes])
            embed_dim = None
            for i, s in enumerate(sizes):
                if s.item() > 0:
                    # Получаем размерность от другого процесса
                    dummy = torch.zeros(1, device=device)
                    if i == rank:
                        if local_embeddings.shape[1] > 0:
                            dummy = torch.tensor([local_embeddings.shape[1]], device=device)
                    dist.broadcast(dummy, src=i)
                    if dummy.item() > 0:
                        embed_dim = int(dummy.item())
                        break
            
            if embed_dim is None:
                embed_dim = 1
            
            padded_embeddings = np.zeros((max_size, embed_dim), dtype=np.float32)
        
        tensor_embeddings = torch.from_numpy(padded_embeddings).to(device)
        gathered = [torch.zeros_like(tensor_embeddings) for _ in range(world_size)]
        dist.all_gather(gathered, tensor_embeddings)
        
        all_embeddings = []
        for i, size in enumerate(sizes):
            if size.item() > 0:
                all_embeddings.append(gathered[i][:size.item()].cpu().numpy())
        
        gathered_paths = [None] * world_size
        dist.all_gather_object(gathered_paths, paths)
        all_paths = []
        for p in gathered_paths:
            all_paths.extend(p)
        
        gathered_broken = [None] * world_size
        dist.all_gather_object(gathered_broken, all_broken_files)
        all_broken_combined = []
        for b in gathered_broken:
            all_broken_combined.extend(b)
        
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        return final_embeddings, all_paths, all_broken_combined
    
    return local_embeddings, paths, all_broken_files


def get_file_sizes(paths):
    """Получаем размеры файлов один раз"""
    sizes = []
    for path in paths:
        try:
            sizes.append(os.path.getsize(path))
        except:
            sizes.append(0)
    return np.array(sizes)


def find_duplicates_faiss(embeddings, paths, threshold, use_gpu=True):
    """
    Поиск дубликатов с FAISS — БЕЗ транзитивности.
    Для каждой пары похожих удаляем файл меньшего размера.
    """
    n, d = embeddings.shape
    logger.info(f"🔍 Finding duplicates using FAISS (n={n:,}, d={d})")
    
    # Получаем размеры файлов заранее
    logger.info("   Getting file sizes...")
    file_sizes = get_file_sizes(paths)
    
    # Нормализуем для косинусного сходства
    logger.info("   Normalizing embeddings...")
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Создаём индекс
    if n < 10000:
        logger.info("   Using exact search (IndexFlatIP)")
        index = faiss.IndexFlatIP(d)
    elif n < 100000:
        nlist = min(int(np.sqrt(n)), 1024)
        logger.info(f"   Using IVF index (nlist={nlist})")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(nlist // 4, 64)
    else:
        nlist = min(int(4 * np.sqrt(n)), 4096)
        logger.info(f"   Using IVF index (nlist={nlist})")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = min(nlist // 4, 128)
    
    # GPU ускорение
    if use_gpu and torch.cuda.is_available():
        try:
            logger.info("   Moving index to GPU...")
            res = faiss.StandardGpuResources()
            res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("   ✅ FAISS using GPU")
        except Exception as e:
            logger.warning(f"   ⚠️ GPU FAISS failed: {e}, using CPU")
    
    # Обучаем если нужно
    if hasattr(index, 'is_trained') and not index.is_trained:
        logger.info("   Training index...")
        train_size = min(n, 100000)
        if train_size < n:
            train_indices = np.random.choice(n, train_size, replace=False)
            index.train(embeddings[train_indices])
        else:
            index.train(embeddings)
    
    # Добавляем векторы
    logger.info("   Adding vectors to index...")
    index.add(embeddings)
    
    # Ищем k ближайших соседей
    k = min(50, n)
    logger.info(f"   Searching for {k} nearest neighbors...")
    similarities, indices = index.search(embeddings, k)
    
    # Множество для удаления (без транзитивности!)
    to_remove = set()
    pairs_found = 0
    
    logger.info("   Processing pairs (no transitivity)...")
    
    for i in tqdm(range(n), desc="   Finding duplicates", leave=False):
        # Если этот файл уже помечен на удаление — пропускаем
        if i in to_remove:
            continue
        
        for k_idx in range(1, k):  # Пропускаем индекс 0 (сам файл)
            j = indices[i, k_idx]
            sim = similarities[i, k_idx]
            
            if j == -1 or j == i:
                continue
            
            # Если сосед уже помечен на удаление — пропускаем
            if j in to_remove:
                continue
            
            if sim >= threshold:
                pairs_found += 1
                
                # Удаляем файл МЕНЬШЕГО размера из пары
                if file_sizes[i] >= file_sizes[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i удалён, переходим к следующему
    
    logger.info(f"   Found {pairs_found:,} duplicate pairs")
    logger.info(f"   Files to remove: {len(to_remove):,}")
    
    return to_remove


def find_duplicates_batched(embeddings, paths, threshold, batch_size=1000):
    """
    Fallback без FAISS — БЕЗ транзитивности.
    """
    n, d = embeddings.shape
    logger.info(f"🔍 Finding duplicates using batched search (n={n:,})")
    logger.warning("   ⚠️ This is O(n²) - consider installing faiss-gpu")
    
    # Получаем размеры файлов
    logger.info("   Getting file sizes...")
    file_sizes = get_file_sizes(paths)
    
    # Нормализуем
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms
    
    to_remove = set()
    pairs_found = 0
    
    num_batches = (n + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="   Processing batches"):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, n)
        
        # Пропускаем уже удалённые
        active_in_batch = [i for i in range(start_i, end_i) if i not in to_remove]
        if not active_in_batch:
            continue
        
        batch = embeddings_norm[active_in_batch]
        
        # Сравниваем со всеми ПОСЛЕ текущего батча (избегаем дублирования пар)
        for j_batch_idx in range(batch_idx, num_batches):
            start_j = j_batch_idx * batch_size
            end_j = min(start_j + batch_size, n)
            
            active_in_other = [j for j in range(start_j, end_j) if j not in to_remove]
            if not active_in_other:
                continue
            
            other_batch = embeddings_norm[active_in_other]
            
            # Косинусное сходство
            sim_matrix = batch @ other_batch.T
            
            # Находим пары выше порога
            for local_i, global_i in enumerate(active_in_batch):
                if global_i in to_remove:
                    continue
                    
                for local_j, global_j in enumerate(active_in_other):
                    if global_j in to_remove:
                        continue
                    
                    # Пропускаем себя и уже обработанные пары
                    if global_i >= global_j:
                        continue
                    
                    if sim_matrix[local_i, local_j] >= threshold:
                        pairs_found += 1
                        
                        # Удаляем меньший файл
                        if file_sizes[global_i] >= file_sizes[global_j]:
                            to_remove.add(global_j)
                        else:
                            to_remove.add(global_i)
                            break  # global_i удалён
    
    logger.info(f"   Found {pairs_found:,} duplicate pairs")
    logger.info(f"   Files to remove: {len(to_remove):,}")
    
    return to_remove


def find_duplicates(embeddings, paths, threshold, use_gpu=True):
    """Выбирает лучший метод"""
    if FAISS_AVAILABLE:
        return find_duplicates_faiss(embeddings, paths, threshold, use_gpu)
    else:
        return find_duplicates_batched(embeddings, paths, threshold)


def delete_files(file_paths, description="files"):
    if not file_paths:
        return 0
    
    unique_paths = list(set(file_paths))
    
    total_size = 0
    existing_files = []
    for path in unique_paths:
        try:
            if os.path.exists(path):
                total_size += os.path.getsize(path)
                existing_files.append(path)
        except:
            pass
    
    if not existing_files:
        return 0
    
    total_size_mb = total_size / (1024 * 1024)
    logger.info(f"🗑️  Deleting {len(existing_files)} {description} ({total_size_mb:.2f} MB)...")
    
    removed_count = 0
    for path in existing_files:
        try:
            os.remove(path)
            removed_count += 1
            if removed_count % 100 == 0:
                logger.info(f"   Deleted {removed_count}/{len(existing_files)} {description}...")
        except Exception as e:
            logger.error(f"Error removing {path}: {e}")
    
    logger.info(f"✅ Successfully deleted {removed_count} {description}")
    return removed_count


def remove_similar_images_worker(rank, world_size, directory, threshold, batch_size, use_dp=False):
    """
    Worker для обработки изображений.
    
    Args:
        rank: Rank процесса (для DDP) или 0 (для DP/single)
        world_size: Количество процессов (для DDP) или 1 (для DP/single)
        directory: Директория с изображениями
        threshold: Порог сходства
        batch_size: Размер батча
        use_dp: Использовать DataParallel вместо DDP
    """
    use_ddp = world_size > 1 and not use_dp
    
    if use_ddp:
        setup_ddp(rank, world_size)
    
    device = f"cuda:{rank}" if (torch.cuda.is_available() and use_ddp) else "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_name = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    model = load_and_compile_model(model_name, device, use_ddp=use_ddp, use_dp=use_dp, rank=rank)
    
    # Получаем data_config
    if use_ddp or use_dp:
        base_model = model.module
    else:
        base_model = model
    
    # Для torch.compile модель может быть обёрнута
    if hasattr(base_model, 'pretrained_cfg'):
        data_config = timm.data.resolve_data_config(base_model.pretrained_cfg)
    elif hasattr(base_model, '_orig_mod') and hasattr(base_model._orig_mod, 'pretrained_cfg'):
        data_config = timm.data.resolve_data_config(base_model._orig_mod.pretrained_cfg)
    else:
        # Fallback
        temp_model = timm.create_model(model_name, pretrained=False)
        data_config = timm.data.resolve_data_config(temp_model.pretrained_cfg)
        del temp_model
    
    transform = timm.data.create_transform(**data_config)
    
    dataset = ImageDataset(directory, transform)
    
    if len(dataset) == 0:
        if rank == 0:
            logger.warning(f"No images found in directory: {directory}")
        if use_ddp:
            cleanup_ddp()
        return
    
    if rank == 0:
        logger.info(f"Found {len(dataset)} images to process")
    
    # Batch size calculation
    if use_ddp:
        per_gpu_batch_size = batch_size // world_size
    elif use_dp:
        # DataParallel автоматически распределяет батч
        per_gpu_batch_size = batch_size
    else:
        per_gpu_batch_size = batch_size
    
    if rank == 0:
        if use_ddp:
            logger.info(f"📊 Using DDP with {world_size} GPUs, batch per GPU: {per_gpu_batch_size}")
        elif use_dp:
            logger.info(f"📊 Using DataParallel with {world_size} GPUs, total batch: {batch_size}")
        else:
            logger.info(f"📊 Single GPU, batch size: {batch_size}")
    
    num_workers = min(32, os.cpu_count() // max(world_size, 1))
    sampler = DistributedSampler(dataset, shuffle=False) if use_ddp else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        sampler=sampler,
        shuffle=False,
        collate_fn=safe_collate_fn
    )
    
    if rank == 0:
        logger.info("Computing embeddings...")
    
    embeddings, paths, broken_files = compute_embeddings_ddp(
        model, dataloader, device, rank, world_size, use_ddp
    )
    
    if use_ddp and rank != 0:
        cleanup_ddp()
        return
    
    logger.info(f"✅ Processed {len(paths)} valid images")
    if broken_files:
        logger.info(f"⚠️  Found {len(broken_files)} broken images")
    
    if len(embeddings) == 0 or len(paths) == 0:
        logger.warning("No valid images to compare!")
        if broken_files:
            delete_files(broken_files, "broken images")
        if use_ddp:
            cleanup_ddp()
        return
    
    # Поиск дубликатов БЕЗ транзитивности
    to_remove_indices = find_duplicates(embeddings, paths, threshold, use_gpu=True)
    
    duplicate_files = [paths[idx] for idx in to_remove_indices]
    
    # Статистика
    logger.info("=" * 60)
    logger.info("📊 SUMMARY:")
    logger.info(f"   Total images scanned: {len(dataset)}")
    logger.info(f"   Valid images: {len(paths)}")
    logger.info(f"   Broken images: {len(broken_files)}")
    logger.info(f"   Duplicates to remove: {len(duplicate_files)}")
    logger.info(f"   Unique images kept: {len(paths) - len(duplicate_files)}")
    logger.info("=" * 60)
    
    if duplicate_files:
        delete_files(duplicate_files, "duplicate images")
    else:
        logger.info("No duplicate images found")
    
    if broken_files:
        delete_files(broken_files, "broken images")
    
    if use_ddp:
        cleanup_ddp()


def remove_similar_images(directory, threshold=0.91, batch_size=128):
    """
    Удаляет похожие изображения из директории.
    Автоматически выбирает режим: DDP (если torchrun), DataParallel (если несколько GPU), или single GPU.
    """
    logger.info(f"Processing directory: {directory}")
    
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Проверяем, запущен ли через torchrun
    if _is_launched_with_torchrun() and world_size > 1:
        logger.info(f"🚀 Detected torchrun launch with {world_size} GPUs, using DDP")
        # DDP уже запущен через mp.spawn или torchrun
        # Этот код не должен достигаться при запуске через torchrun
        # так как каждый процесс уже получает свой rank
        pass
    elif world_size > 1:
        logger.info(f"🚀 Detected {world_size} GPUs, using DataParallel")
        # Обычный запуск с несколькими GPU — используем DataParallel
        remove_similar_images_worker(0, world_size, directory, threshold, batch_size, use_dp=True)
    else:
        logger.info("Using single GPU/CPU")
        remove_similar_images_worker(0, 1, directory, threshold, batch_size, use_dp=False)


def main():
    parser = argparse.ArgumentParser(description='Remove similar images from directories')
    parser.add_argument('--path', type=str, help='Directory path or file with list of directories')
    parser.add_argument('--threshold', type=float, default=0.91, help='Similarity threshold (default: 0.91)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for embeddings')
    parser.add_argument('--use-ddp', action='store_true', help='Force DDP mode with mp.spawn (use only if not using torchrun)')
    
    args = parser.parse_args()
    
    if not args.path:
        logger.error("Please specify --path")
        return
    
    setup_optimizations()
    
    if FAISS_AVAILABLE:
        logger.info("✅ FAISS available - using optimized search")
    else:
        logger.warning("⚠️  FAISS not available - install with: pip install faiss-gpu")
    
    if os.path.isfile(args.path):
        with open(args.path, 'r') as f:
            directories = [line.strip() for line in f if line.strip()]
    else:
        directories = [args.path]
    
    for directory in directories:
        if os.path.isdir(directory):
            logger.info(f"Starting: {directory}")
            try:
                # Если явно указан --use-ddp, используем mp.spawn
                if args.use_ddp:
                    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    if world_size > 1:
                        logger.info(f"🚀 Using DDP with mp.spawn ({world_size} GPUs)")
                        mp.spawn(
                            remove_similar_images_worker,
                            args=(world_size, directory, args.threshold, args.batch_size, False),
                            nprocs=world_size,
                            join=True
                        )
                    else:
                        logger.warning("--use-ddp specified but only 1 GPU available, using single GPU mode")
                        remove_similar_images(directory, args.threshold, args.batch_size)
                else:
                    # Автоматический выбор режима
                    remove_similar_images(directory, args.threshold, args.batch_size)
                
                logger.info(f"Finished: {directory}")
            except Exception as e:
                logger.error(f"Error in {directory}: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Directory not found: {directory}")
    
    logger.info("All done.")


if __name__ == "__main__":
    main()
import os
os.environ["PYTORCH_INDUCTOR_CACHE_DIR"] = "/home/user/cache"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import pillow_jxl
from PIL import Image
import argparse
from tqdm import tqdm

def setup_optimizations():
    """Настройка всех оптимизаций PyTorch"""
    if torch.cuda.is_available():
        # cuDNN оптимизации
        cudnn.enabled = True
        cudnn.benchmark = True  # Автотюнинг для поиска лучших алгоритмов
        cudnn.deterministic = False  # Выключаем детерминированность для производительности
        
        # TF32 для Ampere и новее
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ TF32 enabled")
        
        # PyTorch 2.0 SDPA (Flash Attention, Memory Efficient Attention)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print("✅ PyTorch SDPA (Flash Attention / Memory Efficient) enabled")
        
        # Отключаем детерминированные алгоритмы для скорости
        torch.use_deterministic_algorithms(False)

def setup_ddp(rank, world_size):
    """Инициализация DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Очистка DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()

class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocessor):
        self.image_paths = image_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.preprocessor(images=[image], return_tensors="pt").pixel_values.squeeze()
            return image_path, pixel_values
        except Exception as e:
            # В случае ошибки возвращаем пустое изображение
            print(f"Error loading {image_path}: {e}")
            # Возвращаем черное изображение как fallback
            dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            pixel_values = self.preprocessor(images=[dummy_image], return_tensors="pt").pixel_values.squeeze()
            return image_path, pixel_values

def setup_model(device, use_ddp=False, rank=0):
    """Настройка модели с компиляцией и DDP"""
    if rank == 0 or not use_ddp:
        print("Loading model...")
    
    model, preprocessor = convert_v2_5_from_siglip(
        trust_remote_code=True,
    )
    
    model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()
    
    # torch.compile с max-autotune ВСЕГДА включен
    if rank == 0 or not use_ddp:
        print("🔥 Compiling model with torch.compile(mode='max-autotune')")
    
    try:
        compiled_model = torch.compile(
            model,
            mode="max-autotune",
            fullgraph=False
        )
        
        # Проверка работоспособности
        with torch.inference_mode():
            # Определяем размер входа из preprocessor
            dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            test_input = preprocessor(images=[dummy_image], return_tensors="pt").pixel_values
            test_input = test_input.to(torch.bfloat16).to(device)
            _ = compiled_model(test_input)
        
        if rank == 0 or not use_ddp:
            print("✅ Model compiled successfully with max-autotune")
        model = compiled_model
    except Exception as e:
        if rank == 0 or not use_ddp:
            print(f"⚠️  Compilation failed: {e}")
            print("Using uncompiled model")
    
    # Оборачиваем в DDP если нужно
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)
        if rank == 0:
            print(f"✅ DDP initialized on {dist.get_world_size()} GPUs")
    
    return model, preprocessor

def find_image_paths(directory):
    """Рекурсивный поиск изображений"""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.jxl')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_optimal_worker_count(world_size):
    """Определяем оптимальное количество workers"""
    cpu_count = os.cpu_count()
    # Примерно 4-8 workers на GPU, но не больше чем доступно CPU
    workers_per_gpu = min(8, cpu_count // world_size)
    return max(1, workers_per_gpu)

def process_images_worker(rank, world_size, input_dir, threshold, base_batch_size):
    """Worker функция для DDP обработки"""
    use_ddp = world_size > 1
    
    if use_ddp:
        setup_ddp(rank, world_size)
    
    device = f"cuda:{rank}"
    
    # Настройка модели
    model, preprocessor = setup_model(device, use_ddp, rank)
    
    # Поиск изображений
    image_paths = find_image_paths(input_dir)
    
    if len(image_paths) == 0:
        if rank == 0:
            print(f"No images found in {input_dir}")
        if use_ddp:
            cleanup_ddp()
        return
    
    if rank == 0 or not use_ddp:
        print(f"\nFound {len(image_paths)} images to process")
    
    # Создаем dataset
    dataset = ImageDataset(image_paths, preprocessor)
    
    # Batch size на GPU
    per_gpu_batch_size = base_batch_size // world_size if use_ddp else base_batch_size
    effective_batch_size = per_gpu_batch_size * world_size if use_ddp else base_batch_size
    
    # Оптимальное количество workers
    num_workers = get_optimal_worker_count(world_size)
    
    if rank == 0 or not use_ddp:
        if use_ddp:
            print(f"🚀 Using {world_size} GPUs")
            print(f"📊 Effective batch size: {effective_batch_size} (per GPU: {per_gpu_batch_size})")
        else:
            print(f"📊 Batch size: {base_batch_size}")
        print(f"👷 Number of workers: {num_workers}")
    
    # Создаем sampler для DDP
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if use_ddp else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Обработка изображений
    local_deleted_count = 0
    local_kept_count = 0
    local_results = []  # Для синхронизации результатов
    
    with torch.inference_mode():
        for batch_paths, pixel_values in tqdm(dataloader, 
                                             desc=f"GPU {rank}" if use_ddp else "Processing",
                                             disable=use_ddp and rank != 0):
            pixel_values = pixel_values.to(torch.bfloat16).to(device, non_blocking=True)
            
            scores = model(pixel_values).logits.squeeze()
            
            # Обрабатываем случай единичного батча
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            
            scores = scores.float().cpu().numpy()
            
            for path, score in zip(batch_paths, scores):
                local_results.append((path, score))
    
    # Собираем результаты со всех GPU
    if use_ddp:
        # Синхронизируем все процессы
        dist.barrier()
        
        # Собираем результаты
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, local_results)
        
        # Только rank 0 обрабатывает результаты
        if rank == 0:
            all_results = []
            for results in gathered_results:
                all_results.extend(results)
        else:
            cleanup_ddp()
            return
    else:
        all_results = local_results
    
    # Удаляем файлы (только rank 0)
    deleted_count = 0
    kept_count = 0
    
    print("\n" + "="*80)
    print("Processing results:")
    print("="*80)
    
    for path, score in all_results:
        filename = os.path.basename(path)
        if score < threshold:
            try:
                os.remove(path)
                deleted_count += 1
                print(f"❌ Deleted {filename}: Score {score:.2f}")
            except Exception as e:
                print(f"⚠️  Error deleting {filename}: {e}")
        else:
            kept_count += 1
            print(f"✅ Kept {filename}: Score {score:.2f}")
    
    print("\n" + "="*80)
    print("Processing complete:")
    print("="*80)
    print(f"Total images processed: {len(all_results)}")
    print(f"Images deleted: {deleted_count}")
    print(f"Images kept: {kept_count}")
    print(f"Deletion rate: {deleted_count/len(all_results)*100:.1f}%")
    print("="*80)
    
    if use_ddp:
        cleanup_ddp()

def main(input_dir, threshold, batch_size):
    """Основная функция с автоматическим определением multi-GPU"""
    # Выводим информацию о системе
    print("="*80)
    print("System Information:")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*80)
    
    # Настраиваем все оптимизации
    setup_optimizations()
    
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size == 0:
        raise RuntimeError("No GPU devices found!")
    
    if world_size > 1:
        # Multi-GPU с DDP
        print(f"\n🚀 Launching DDP with {world_size} GPUs\n")
        mp.spawn(
            process_images_worker,
            args=(world_size, input_dir, threshold, batch_size),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU
        print(f"\n🎯 Using single GPU mode\n")
        process_images_worker(0, 1, input_dir, threshold, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images and delete those below a certain aesthetic score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Path to the directory containing images")
    parser.add_argument("--threshold", type=float, default=4.2, 
                       help="Threshold score for deletion")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Total batch size (will be divided by number of GPUs)")
    
    args = parser.parse_args()
    
    # Проверяем существование директории
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist!")
        exit(1)
    
    main(args.input_dir, args.threshold, args.batch_size)
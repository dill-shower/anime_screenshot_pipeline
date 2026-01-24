import os
import logging
import subprocess
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import json

from .basics import parse_anime_info
from .remove_duplicates import DuplicateRemover


def find_best_allocator():
    """Find the best available memory allocator."""
    allocators = [
        ('/usr/lib/x86_64-linux-gnu/libmimalloc.so.2', 'mimalloc'),
        ('/usr/lib/x86_64-linux-gnu/libmimalloc.so', 'mimalloc'),
        ('/usr/local/lib/libmimalloc.so', 'mimalloc'),
        ('/usr/lib/x86_64-linux-gnu/libjemalloc.so.2', 'jemalloc'),
        ('/usr/lib/x86_64-linux-gnu/libjemalloc.so', 'jemalloc'),
        ('/usr/local/lib/libjemalloc.so', 'jemalloc'),
        ('/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4', 'tcmalloc'),
        ('/usr/lib/x86_64-linux-gnu/libtcmalloc.so', 'tcmalloc'),
    ]
    
    for lib_path, name in allocators:
        if os.path.exists(lib_path):
            return lib_path, name
    
    return None, 'system'


def get_physical_cpu_count():
    """Determine the number of PHYSICAL CPU cores."""
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores:
            return physical_cores
    except ImportError:
        pass
    
    try:
        lscpu_output = subprocess.check_output(['lscpu'], universal_newlines=True)
        cores_per_socket = None
        sockets = None
        
        for line in lscpu_output.split('\n'):
            if 'Core(s) per socket:' in line:
                cores_per_socket = int(line.split(':')[1].strip())
            elif 'Socket(s):' in line:
                sockets = int(line.split(':')[1].strip())
        
        if cores_per_socket and sockets:
            return cores_per_socket * sockets
    except Exception:
        pass
    
    logical_cores = os.cpu_count()
    if logical_cores:
        return max(1, logical_cores // 2)
    
    return 1


def get_optimal_config_for_massive_cpu():
    """Optimization for MAXIMUM speed."""
    logical_cores = os.cpu_count()
    physical_cores = get_physical_cpu_count()
    
    logging.info(f"CPU Detection:")
    logging.info(f"  Logical cores (with SMT): {logical_cores}")
    logging.info(f"  Physical cores: {physical_cores}")
    logging.info(f"  SMT ratio: {logical_cores / physical_cores:.1f}x")
    
    target_logical_usage = int(logical_cores * 0.75)
    optimal_threads_per_ffmpeg = 20
    base_num_processes = max(1, target_logical_usage // optimal_threads_per_ffmpeg)
    num_parts = max(4, min(16, base_num_processes))
    threads_per_part = max(12, target_logical_usage // num_parts)
    
    if threads_per_part > 24:
        threads_per_part = 24
    
    total_threads = num_parts * threads_per_part
    
    return {
        'num_parts': num_parts,
        'threads_per_part': threads_per_part,
        'total_threads': total_threads,
        'logical_cores': logical_cores,
        'physical_cores': physical_cores,
        'smt_ratio': logical_cores / physical_cores if physical_cores > 0 else 1,
        'target_usage': target_logical_usage,
        'actual_usage_pct': (total_threads / logical_cores * 100) if logical_cores > 0 else 0,
    }


def get_video_info(video_path: str) -> Tuple[float, float, int]:
    """Get video information."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration,r_frame_rate,nb_frames:format=duration',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    
    stream = data['streams'][0]
    
    duration = stream.get('duration')
    if duration is None and 'format' in data:
        duration = data['format'].get('duration')
    if duration is None:
        raise ValueError("Could not determine video duration")
    
    duration = float(duration)
    
    fps_str = stream['r_frame_rate']
    num, den = map(int, fps_str.split('/'))
    fps = num / den
    
    nb_frames = stream.get('nb_frames')
    if nb_frames:
        nb_frames = int(nb_frames)
    else:
        nb_frames = int(duration * fps)
    
    return duration, fps, nb_frames


def adjust_parts_for_duration(base_config: dict, duration: float, logger: logging.Logger) -> dict:
    """Adjust the number of parts based on video duration."""
    base_num_parts = base_config['num_parts']
    min_part_duration = 90
    
    max_parts_for_video = max(1, int(duration / min_part_duration))
    actual_num_parts = min(base_num_parts, max_parts_for_video)
    
    target_usage = base_config['target_usage']
    threads_per_part = max(12, target_usage // actual_num_parts)
    
    if threads_per_part > 24:
        threads_per_part = 24
    
    adjusted = {
        'num_parts': actual_num_parts,
        'threads_per_part': threads_per_part,
        'total_threads': actual_num_parts * threads_per_part,
        'logical_cores': base_config['logical_cores'],
        'physical_cores': base_config['physical_cores'],
        'smt_ratio': base_config['smt_ratio'],
        'target_usage': base_config['target_usage'],
        'actual_usage_pct': (actual_num_parts * threads_per_part / base_config['logical_cores'] * 100),
        'part_duration': duration / actual_num_parts
    }
    
    if actual_num_parts != base_num_parts:
        logger.info(f"Adjusted parts: {base_num_parts} -> {actual_num_parts} (video duration)")
        logger.info(f"   Part duration: {adjusted['part_duration']:.1f}s (min: {min_part_duration}s)")
    
    return adjusted


def calculate_split_points(video_path: str, num_parts: int, logger: logging.Logger) -> List[Tuple[float, float]]:
    """Calculate video split points."""
    duration, fps, nb_frames = get_video_info(video_path)
    
    logger.info(f"Video info:")
    logger.info(f"   Duration: {duration:.2f}s ({duration/60:.1f} min)")
    logger.info(f"   FPS: {fps:.2f}")
    logger.info(f"   Frames: ~{nb_frames:,}")
    
    if num_parts <= 1:
        return [(0, duration)]
    
    part_duration = duration / num_parts
    split_points = []
    
    for i in range(num_parts):
        start = i * part_duration
        if i == num_parts - 1:
            dur = duration - start
        else:
            dur = part_duration
        split_points.append((start, dur))
    
    logger.info(f"Splitting into {num_parts} parts:")
    logger.info(f"   Part duration: ~{part_duration:.1f}s ({part_duration/60:.1f} min)")
    if num_parts <= 10:
        for i, (start, dur) in enumerate(split_points):
            logger.info(f"   Part {i}: {start/60:.1f}min -> {(start+dur)/60:.1f}min")
    
    return split_points


def process_video_part(args):
    """Process one video part - extract frames to JPEG XL."""
    (input_file, part_id, start_time, duration, output_dir, 
     prefix_anime, anime_name, ep_num, extract_key, threads, 
     allocator_lib, jxl_effort) = args
    
    part_dir = os.path.join(output_dir, f'_temp_part_{part_id:04d}')
    os.makedirs(part_dir, exist_ok=True)
    
    output_pattern = os.path.join(part_dir, f"frame_%08d.jxl")
    
    command = [
        "ffmpeg",
        "-y",
        "-threads", str(threads),
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", input_file,
        "-an",
    ]

    if extract_key:
        command.extend([
            "-vf", "select='eq(pict_type\\,I)'",
            "-vsync", "vfr"
        ])
    else:
        command.extend([
            "-filter:v",
            "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB"
        ])

    # JPEG XL lossless with minimal effort for maximum speed
    command.extend([
        "-c:v", "libjxl",
        "-modular", "1",
        "-distance", "0",           # 0 = lossless
        "-effort", str(jxl_effort), # 1 = fastest, 9 = slowest/best
        "-loglevel", "error",
        output_pattern
    ])
    
    try:
        env = os.environ.copy()
        
        if allocator_lib:
            env['LD_PRELOAD'] = allocator_lib
            if 'mimalloc' in allocator_lib:
                env['MIMALLOC_LARGE_OS_PAGES'] = '1'
                env['MIMALLOC_EAGER_COMMIT'] = '1'
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        stderr_output = []
        for line in process.stderr:
            stderr_output.append(line)
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = ''.join(stderr_output[-10:])
            raise subprocess.CalledProcessError(process.returncode, command, error_msg)
        
        frames = list(Path(part_dir).glob("frame_*.jxl"))
        if not frames:
            raise Exception(f"No frames extracted for part {part_id}")
        
        return {
            'part_id': part_id,
            'part_dir': part_dir,
            'frame_count': len(frames),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        if os.path.exists(part_dir):
            shutil.rmtree(part_dir, ignore_errors=True)
        
        return {
            'part_id': part_id,
            'part_dir': None,
            'frame_count': 0,
            'success': False,
            'error': str(e)
        }


def merge_frames_fast(results: List[dict], output_dir: str, prefix_anime: str, 
                      anime_name: str, ep_num: int, logger: logging.Logger) -> str:
    """Fast frame merging."""
    results = sorted(results, key=lambda x: x['part_id'])
    
    folder_name = f"{prefix_anime}_{anime_name}_EP{ep_num}"
    final_dir = os.path.join(output_dir, folder_name)
    os.makedirs(final_dir, exist_ok=True)
    
    global_frame_counter = 1
    total_frames = 0
    
    logger.info(f"Merging frames from {len(results)} parts...")
    
    successful_results = [r for r in results if r['success']]
    
    for idx, result in enumerate(successful_results):
        part_dir = result['part_dir']
        frames = sorted(Path(part_dir).glob("frame_*.jxl"))
        
        for frame_path in frames:
            new_name = f"{prefix_anime}_{anime_name}_EP{ep_num}_{global_frame_counter:08d}.jxl"
            new_path = os.path.join(final_dir, new_name)
            os.rename(str(frame_path), new_path)
            global_frame_counter += 1
        
        total_frames += len(frames)
        
        try:
            os.rmdir(part_dir)
        except Exception:
            shutil.rmtree(part_dir, ignore_errors=True)
    
    logger.info(f"Merged {total_frames:,} frames")
    return final_dir


def process_single_video(
    video_file: str,
    dst_dir: str,
    prefix_anime: str,
    anime_name: str,
    ep_num: int,
    extract_key: bool,
    duplicate_remover: Optional[DuplicateRemover],
    logger: logging.Logger,
    jxl_effort: int,
    allocator_lib: Optional[str],
    base_config: dict,
) -> None:
    """Process a single video file."""
    import time
    
    logger.info(f"Processing: {os.path.basename(video_file)}")
    logger.info(f"   Anime: {anime_name}")
    logger.info(f"   Episode: {ep_num}\n")
    
    duration, fps, nb_frames = get_video_info(video_file)
    config = adjust_parts_for_duration(base_config, duration, logger)
    
    logger.info(f"Processing Configuration:")
    logger.info(f"   Video parts: {config['num_parts']}")
    logger.info(f"   Threads per part: {config['threads_per_part']}")
    logger.info(f"   Total threads: {config['total_threads']}")
    logger.info(f"   CPU utilization: ~{config['actual_usage_pct']:.0f}%")
    logger.info(f"   Part duration: ~{config['part_duration']:.1f}s ({config['part_duration']/60:.1f} min)\n")
    
    split_points = calculate_split_points(video_file, config['num_parts'], logger)
    
    temp_base_dir = os.path.join(dst_dir, f"_temp_ep{ep_num}")
    os.makedirs(temp_base_dir, exist_ok=True)
    
    tasks = []
    for part_id, (start_time, part_duration) in enumerate(split_points):
        tasks.append((
            video_file,
            part_id,
            start_time,
            part_duration,
            temp_base_dir,
            prefix_anime,
            anime_name,
            ep_num,
            extract_key,
            config['threads_per_part'],
            allocator_lib,
            jxl_effort,
        ))
    
    start_time_total = time.time()
    
    logger.info(f"\nStarting frame extraction (JPEG XL lossless)...\n")
    
    results = []
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=config['num_parts']) as executor:
        future_to_part = {
            executor.submit(process_video_part, task): task[1]
            for task in tasks
        }
        
        for future in as_completed(future_to_part):
            part_id = future_to_part[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    completed += 1
                    if completed == 1 or completed % max(1, config['num_parts'] // 4) == 0 or completed == config['num_parts']:
                        elapsed = time.time() - start_time_total
                        eta = (elapsed / completed) * (config['num_parts'] - completed) if completed > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{config['num_parts']} "
                            f"({completed/config['num_parts']*100:.0f}%) "
                            f"| {elapsed:.0f}s elapsed, ETA: {eta:.0f}s"
                        )
                else:
                    failed += 1
                    logger.error(f"Part {part_id} failed: {result['error']}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"Part {part_id} error: {str(e)}")
    
    processing_time = time.time() - start_time_total
    
    successful_parts = [r for r in results if r['success']]
    if not successful_parts:
        logger.error(f"All parts failed!")
        shutil.rmtree(temp_base_dir, ignore_errors=True)
        return
    
    total_frames = sum(r['frame_count'] for r in successful_parts)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Frame extraction completed:")
    logger.info(f"  Successful: {len(successful_parts)}/{config['num_parts']}")
    logger.info(f"  Failed: {failed}/{config['num_parts']}")
    logger.info(f"  Total frames: {total_frames:,}")
    logger.info(f"  Time: {processing_time:.2f}s ({processing_time/60:.2f} min)")
    logger.info(f"  Throughput: {total_frames/processing_time:.1f} frames/sec")
    logger.info(f"{'='*70}\n")
    
    merge_start = time.time()
    final_dir = merge_frames_fast(results, dst_dir, prefix_anime, anime_name, ep_num, logger)
    merge_time = time.time() - merge_start
    logger.info(f"Merge time: {merge_time:.2f}s\n")
    
    try:
        os.rmdir(temp_base_dir)
    except Exception:
        shutil.rmtree(temp_base_dir, ignore_errors=True)
    
    actual_size_bytes = sum(f.stat().st_size for f in Path(final_dir).glob("*.jxl"))
    actual_size_gb = actual_size_bytes / (1024**3)
    
    if duplicate_remover is not None:
        logger.info(f"Removing duplicates...")
        dup_start = time.time()
        try:
            duplicate_remover.remove_similar_from_dir(final_dir)
            dup_time = time.time() - dup_start
            
            final_size_bytes = sum(f.stat().st_size for f in Path(final_dir).glob("*.jxl"))
            final_size_gb = final_size_bytes / (1024**3)
            removed_gb = actual_size_gb - final_size_gb
            
            logger.info(f"Duplicates removed in {dup_time:.2f}s")
            logger.info(f"   Freed space: {removed_gb:.2f} GB\n")
        except Exception as e:
            logger.error(f"Duplicate removal error: {str(e)}")
            final_size_gb = actual_size_gb
    else:
        final_size_gb = actual_size_gb
    
    total_time = time.time() - start_time_total
    final_frame_count = len(list(Path(final_dir).glob("*.jxl")))
    
    logger.info(f"{'='*70}")
    logger.info(f"EXTRACTION COMPLETED")
    logger.info(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    logger.info(f"   Format: JPEG XL lossless (effort={jxl_effort})")
    logger.info(f"   Output size: {final_size_gb:.2f} GB ({final_frame_count:,} frames)")
    if final_frame_count > 0:
        avg_size_kb = (actual_size_bytes / 1024) / final_frame_count
        logger.info(f"   Avg per frame: {avg_size_kb:.0f} KB")
    logger.info(f"   Output: {final_dir}")
    logger.info(f"{'='*70}\n")


def extract_and_remove_similar(
    src_path: str,
    dst_dir: str,
    prefix: Optional[str] = None,
    ep_init: Optional[int] = None,
    extract_key: bool = False,
    duplicate_remover: Optional[DuplicateRemover] = None,
    logger: Optional[logging.Logger] = None,
    jxl_effort: int = 1,
) -> None:
    """
    Extract frames from video in JPEG XL format (lossless).
    
    Supports both a directory with video files and a single file.
    
    Args:
        src_path: path to video file OR directory with videos
        dst_dir: output directory
        prefix: prefix for file names
        ep_init: starting episode number
        extract_key: extract only key frames
        duplicate_remover: duplicate remover instance
        logger: logger instance
        jxl_effort: 1-9 (1 = fastest, 9 = best compression)
    """
    if logger is None:
        logger = logging.getLogger()

    video_extensions = {'.mp4', '.mkv', '.avi', '.flv', '.mov', '.wmv', '.ts', '.m2ts', '.webm'}

    # ============================================================
    # If a directory is passed - find all videos and process each
    # ============================================================
    if os.path.isdir(src_path):
        video_files = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(src_path)
            for file in files
            if Path(file).suffix.lower() in video_extensions
        ])
        
        if not video_files:
            logger.error(f"No video files found in: {src_path}")
            return
        
        logger.info(f"{'='*70}")
        logger.info(f"Found {len(video_files)} video files in directory:")
        for vf in video_files:
            logger.info(f"   - {os.path.basename(vf)}")
        logger.info(f"{'='*70}\n")
        
        # Initialize once for all files
        allocator_lib, allocator_name = find_best_allocator()
        
        logger.info(f"{'='*70}")
        logger.info(f"Memory Allocator:")
        if allocator_lib:
            logger.info(f"   Using: {allocator_name} ({allocator_lib})")
        else:
            logger.info(f"   Using: system default (glibc malloc)")
            logger.info(f"   Tip: Install mimalloc for +15-20% speedup:")
            logger.info(f"      sudo apt-get install libmimalloc2.0")
        logger.info(f"{'='*70}\n")

        base_config = get_optimal_config_for_massive_cpu()
        
        logger.info(f"{'='*70}")
        logger.info(f"CPU Configuration:")
        logger.info(f"   Physical cores: {base_config['physical_cores']}")
        logger.info(f"   Logical cores (SMT): {base_config['logical_cores']}")
        logger.info(f"   SMT ratio: {base_config['smt_ratio']:.1f}x")
        logger.info(f"   Target CPU usage: {base_config['target_usage']} threads (~{base_config['actual_usage_pct']:.0f}%)")
        logger.info(f"   Base parts: {base_config['num_parts']}")
        logger.info(f"   Threads per part: {base_config['threads_per_part']}")
        logger.info(f"{'='*70}\n")
        
        logger.info(f"Output format: JPEG XL lossless (effort={jxl_effort})")
        logger.info(f"   effort 1 = fastest encoding")
        logger.info(f"   effort 9 = best compression (slower)\n")
        
        # Process each file
        for i, video_file in enumerate(video_files):
            ep_num = (ep_init + i) if ep_init is not None else i + 1
            
            filename_without_ext = Path(video_file).stem
            anime_name, _ = parse_anime_info(filename_without_ext)
            prefix_anime = prefix if isinstance(prefix, str) else ""
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing file {i+1}/{len(video_files)}")
            logger.info(f"{'='*70}\n")
            
            try:
                process_single_video(
                    video_file=video_file,
                    dst_dir=dst_dir,
                    prefix_anime=prefix_anime,
                    anime_name=anime_name,
                    ep_num=ep_num,
                    extract_key=extract_key,
                    duplicate_remover=duplicate_remover,
                    logger=logger,
                    jxl_effort=jxl_effort,
                    allocator_lib=allocator_lib,
                    base_config=base_config,
                )
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"ALL {len(video_files)} FILES PROCESSED")
        logger.info(f"{'='*70}\n")
        return

    # ============================================================
    # Process a single file
    # ============================================================
    if not os.path.exists(src_path):
        logger.error(f"File not found: {src_path}")
        return
    
    if Path(src_path).suffix.lower() not in video_extensions:
        logger.error(f"Not a video file: {src_path}")
        return

    allocator_lib, allocator_name = find_best_allocator()
    
    logger.info(f"{'='*70}")
    logger.info(f"Memory Allocator:")
    if allocator_lib:
        logger.info(f"   Using: {allocator_name} ({allocator_lib})")
    else:
        logger.info(f"   Using: system default (glibc malloc)")
        logger.info(f"   Tip: Install mimalloc for +15-20% speedup:")
        logger.info(f"      sudo apt-get install libmimalloc2.0")
    logger.info(f"{'='*70}\n")

    base_config = get_optimal_config_for_massive_cpu()
    
    logger.info(f"{'='*70}")
    logger.info(f"CPU Configuration:")
    logger.info(f"   Physical cores: {base_config['physical_cores']}")
    logger.info(f"   Logical cores (SMT): {base_config['logical_cores']}")
    logger.info(f"   SMT ratio: {base_config['smt_ratio']:.1f}x")
    logger.info(f"   Target CPU usage: {base_config['target_usage']} threads (~{base_config['actual_usage_pct']:.0f}%)")
    logger.info(f"   Base parts: {base_config['num_parts']}")
    logger.info(f"   Threads per part: {base_config['threads_per_part']}")
    logger.info(f"{'='*70}\n")
    
    logger.info(f"Output format: JPEG XL lossless (effort={jxl_effort})")
    logger.info(f"   effort 1 = fastest encoding")
    logger.info(f"   effort 9 = best compression (slower)\n")

    video_file = src_path
    filename_without_ext = Path(video_file).stem
    anime_name, _ = parse_anime_info(filename_without_ext)
    prefix_anime = prefix if isinstance(prefix, str) else ""
    ep_num = ep_init if ep_init is not None else 1

    try:
        process_single_video(
            video_file=video_file,
            dst_dir=dst_dir,
            prefix_anime=prefix_anime,
            anime_name=anime_name,
            ep_num=ep_num,
            extract_key=extract_key,
            duplicate_remover=duplicate_remover,
            logger=logger,
            jxl_effort=jxl_effort,
            allocator_lib=allocator_lib,
            base_config=base_config,
        )
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Works with both directory and single file:
    extract_and_remove_similar(
        src_path="./videos/",  # or "video.mp4"
        dst_dir="./output",
        prefix="anime",
        ep_init=1,
        extract_key=False,
        jxl_effort=1,
        logger=logger
    )

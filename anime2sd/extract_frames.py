import os
import re
import logging
import subprocess
import platform
from typing import Optional

from .basics import parse_anime_info
from .remove_duplicates import DuplicateRemover


def get_physical_core_count(logger=None):
    """Get the number of physical CPU cores (not logical/hyperthreaded)."""
    if logger is None:
        logger = logging.getLogger()
    
    # Try psutil first (most reliable cross-platform method)
    try:
        import psutil
        count = psutil.cpu_count(logical=False)
        if count:
            logger.info(f"Detected {count} physical cores via psutil")
            return count
    except ImportError:
        pass
    
    # Try reading /proc/cpuinfo on Linux
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
            
            # Count unique physical id + core id combinations
            physical_cores = set()
            physical_id = None
            core_id = None
            
            for line in content.split('\n'):
                if line.startswith('physical id'):
                    physical_id = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    core_id = line.split(':')[1].strip()
                elif line.strip() == '' and physical_id is not None and core_id is not None:
                    physical_cores.add((physical_id, core_id))
                    physical_id = None
                    core_id = None
            
            if physical_id is not None and core_id is not None:
                physical_cores.add((physical_id, core_id))
            
            if physical_cores:
                count = len(physical_cores)
                logger.info(f"Detected {count} physical cores via /proc/cpuinfo")
                return count
            
            # Fallback: use cpu cores field
            cpu_cores_matches = re.findall(r'cpu cores\s*:\s*(\d+)', content)
            if cpu_cores_matches:
                cores_per_socket = int(cpu_cores_matches[0])
                physical_ids = set(re.findall(r'physical id\s*:\s*(\d+)', content))
                num_sockets = len(physical_ids) if physical_ids else 1
                count = cores_per_socket * num_sockets
                logger.info(f"Detected {count} physical cores via cpu cores field")
                return count
    except Exception as e:
        logger.debug(f"Failed to read /proc/cpuinfo: {e}")
    
    # Try lscpu on Linux
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            cores_match = re.search(r'Core\(s\) per socket:\s*(\d+)', output)
            sockets_match = re.search(r'Socket\(s\):\s*(\d+)', output)
            if cores_match and sockets_match:
                count = int(cores_match.group(1)) * int(sockets_match.group(1))
                logger.info(f"Detected {count} physical cores via lscpu")
                return count
    except Exception as e:
        logger.debug(f"Failed to run lscpu: {e}")
    
    # Try wmic on Windows
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'NumberOfCores'],
                capture_output=True, text=True, shell=True
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.split('\n') 
                        if l.strip() and l.strip() != 'NumberOfCores']
                if lines:
                    count = sum(int(l) for l in lines if l.isdigit())
                    if count:
                        logger.info(f"Detected {count} physical cores via wmic")
                        return count
        except Exception as e:
            logger.debug(f"Failed to run wmic: {e}")
    
    # Fallback: assume hyperthreading with 2 threads per core
    logical_count = os.cpu_count()
    if logical_count:
        count = max(1, logical_count // 2)
        logger.warning(f"Could not detect physical cores, assuming {count} (half of {logical_count} logical)")
        return count
    
    logger.warning("Could not detect CPU cores, defaulting to 1")
    return 1


def find_mimalloc(logger=None):
    """Find mimalloc library path if available (Linux only)."""
    if logger is None:
        logger = logging.getLogger()
    
    # mimalloc LD_PRELOAD only works on Linux
    if platform.system() != 'Linux':
        logger.debug("mimalloc preloading is only supported on Linux")
        return None
    
    # Common paths for mimalloc
    possible_paths = [
        '/usr/lib/libmimalloc.so',
        '/usr/lib/libmimalloc.so.2',
        '/usr/lib/libmimalloc.so.1',
        '/usr/local/lib/libmimalloc.so',
        '/usr/local/lib/libmimalloc.so.2',
        '/usr/local/lib/libmimalloc.so.1',
        '/usr/lib/x86_64-linux-gnu/libmimalloc.so',
        '/usr/lib/x86_64-linux-gnu/libmimalloc.so.2',
        '/usr/lib/x86_64-linux-gnu/libmimalloc.so.1',
        '/usr/lib64/libmimalloc.so',
        '/usr/lib64/libmimalloc.so.2',
        '/usr/lib64/libmimalloc.so.1',
    ]
    
    # Check common paths first
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found mimalloc at {path}")
            return path
    
    # Try ldconfig
    try:
        result = subprocess.run(
            ['ldconfig', '-p'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'libmimalloc' in line:
                    match = re.search(r'=>\s*(.+)$', line)
                    if match:
                        path = match.group(1).strip()
                        if os.path.exists(path):
                            logger.info(f"Found mimalloc via ldconfig at {path}")
                            return path
    except Exception as e:
        logger.debug(f"Failed to run ldconfig: {e}")
    
    logger.debug("mimalloc not found in system")
    return None


def get_ffmpeg_command(file, file_pattern, extract_key, num_threads, logger=None):
    """Build FFmpeg command for CPU-based frame extraction to BMP format."""
    if logger is None:
        logger = logging.getLogger()
    
    command = ["ffmpeg"]
    
    # Set number of threads for decoding
    command.extend(["-threads", str(num_threads)])
    
    command.extend(["-i", file])

    if extract_key:
        command.extend(["-vf", "select='eq(pict_type,I)'", "-vsync", "vfr"])
    else:
        command.extend(
            [
                "-filter:v",
                "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB",
            ]
        )

    # Set threads for encoding/output as well
    command.extend(["-threads", str(num_threads)])
    
    # BMP is inherently lossless and uncompressed, no quality params needed
    command.append(file_pattern)

    return command


def extract_and_remove_similar(
    src_dir: str,
    dst_dir: str,
    prefix: Optional[str] = None,
    ep_init: Optional[int] = None,
    extract_key: bool = False,
    duplicate_remover: Optional[DuplicateRemover] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Extracts frames from video files in the specified source directory,
    saves them to the destination directory, and optionally removes similar frames.

    The function supports multiple video file formats such as mp4, mkv, avi, etc.
    It uses FFmpeg to extract frames from videos.
    If a `DuplicateRemover` instance is provided, it removes similar frames within each
    episode's directory and across the entire source directory (for opening and ending).

    Args:
        src_dir (str):
            The directory containing source video files.
        dst_dir (str):
            The directory where extracted frames will be saved.
        prefix (Optional[str]):
            A prefix to add to the names of extracted frames.
            Defaults to None in which case prefix if inferred from file name.
        ep_init (Optional[int]):
            An initial episode number to start from for naming the extracted frames.
            Defaults to None in which case episode number is inferred from file name.
        extract_key (bool):
            Flag indicating whether to extract only key frames.
            Defaults to False.
        duplicate_remover (Optional[DuplicateRemover]):
            An instance of DuplicateRemover to remove duplicate frames.
            Defaults to None in which case no duplicate removal is performed.
        logger (Optional[logging.Logger]):
            A logger for logging messages.
            Defaults to None in which case a default logger is used.
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Get physical core count for threading
    num_threads = get_physical_core_count(logger)
    logger.info(f"Using {num_threads} threads for FFmpeg (physical cores)")
    
    # Check for mimalloc and prepare environment
    mimalloc_path = find_mimalloc(logger)
    env = os.environ.copy()
    if mimalloc_path:
        logger.info(f"Using mimalloc for memory allocation: {mimalloc_path}")
        # Prepend to existing LD_PRELOAD if any
        existing_preload = env.get('LD_PRELOAD', '')
        if existing_preload:
            env['LD_PRELOAD'] = f"{mimalloc_path}:{existing_preload}"
        else:
            env['LD_PRELOAD'] = mimalloc_path
    
    # Supported video file extensions
    video_extensions = [".mp4", ".mkv", ".avi", ".flv", ".mov", ".wmv", ".ts", ".m2ts"]

    # Recursively find all video files in the specified
    # source directory and its subdirectories
    files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(src_dir)
        for file in files
        if os.path.splitext(file)[1] in video_extensions
    ]

    # Loop through each file
    for i, file in enumerate(sorted(files)):
        # Extract the filename without extension
        filename_without_ext = os.path.splitext(os.path.basename(file))[0]

        # Extract the anime name and episode number
        anime_name, ep_num = parse_anime_info(filename_without_ext)
        anime_name = "_".join(re.split(r"\s+", anime_name))
        prefix_anime = f"{prefix if isinstance(prefix, str) else anime_name}_"
        if isinstance(ep_init, int):
            ep_num = i + ep_init
        elif ep_num is None:
            ep_num = i

        # Create the output directory
        dst_ep_dir = os.path.join(dst_dir, filename_without_ext)
        os.makedirs(dst_ep_dir, exist_ok=True)
        # BMP format for lossless uncompressed output
        file_pattern = os.path.join(dst_ep_dir, f"{prefix_anime}EP{ep_num}_%d.bmp")

        # Run ffmpeg on the file, saving the output to the output directory
        ffmpeg_command = get_ffmpeg_command(file, file_pattern, extract_key, num_threads, logger)
        logger.info(ffmpeg_command)
        subprocess.run(ffmpeg_command, check=True, env=env)

        if duplicate_remover is not None:
            duplicate_remover.remove_similar_from_dir(dst_ep_dir)

    # Go through all files again to remove duplicates from op and ed
    if duplicate_remover is not None:
        duplicate_remover.remove_similar_from_dir(dst_dir, portion="first")
        duplicate_remover.remove_similar_from_dir(dst_dir, portion="last")

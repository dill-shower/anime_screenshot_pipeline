import os
import logging
from tqdm import tqdm
from typing import List, Tuple, Set, Optional, Dict
from PIL import Image, ImageFilter
from pathlib import Path
import subprocess
import time
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .basics import get_related_paths, get_images_recursively


# Optional FAISS (recommended). If absent, code falls back to slower numpy chunking.
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


def get_physical_core_count(logger: Optional[logging.Logger] = None) -> int:
    """
    Linux: try /sys topology (best), fallback to /proc/cpuinfo, then os.cpu_count().
    Returns physical cores (not logical threads).
    """
    if logger is None:
        logger = logging.getLogger()

    # 1) /sys topology: count unique (package, core_id)
    try:
        topo_root = "/sys/devices/system/cpu"
        cores = set()
        if os.path.isdir(topo_root):
            for name in os.listdir(topo_root):
                if not name.startswith("cpu"):
                    continue
                suffix = name[3:]
                if not suffix.isdigit():
                    continue
                cpu_dir = os.path.join(topo_root, name, "topology")
                core_id_path = os.path.join(cpu_dir, "core_id")
                pkg_id_path = os.path.join(cpu_dir, "physical_package_id")
                if os.path.exists(core_id_path) and os.path.exists(pkg_id_path):
                    with open(core_id_path, "r") as f:
                        core_id = f.read().strip()
                    with open(pkg_id_path, "r") as f:
                        pkg_id = f.read().strip()
                    cores.add((pkg_id, core_id))
        if cores:
            return len(cores)
    except Exception as e:
        logger.debug(f"Failed to detect physical cores via /sys topology: {e}")

    # 2) /proc/cpuinfo fallback: unique (physical id, core id)
    try:
        if os.path.exists("/proc/cpuinfo"):
            physical_cores = set()
            physical_id = None
            core_id = None
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("physical id"):
                        physical_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    elif line == "":
                        if physical_id is not None and core_id is not None:
                            physical_cores.add((physical_id, core_id))
                        physical_id = None
                        core_id = None
            if physical_cores:
                return len(physical_cores)
    except Exception as e:
        logger.debug(f"Failed to detect physical cores via /proc/cpuinfo: {e}")

    # 3) last resort
    return max(1, os.cpu_count() or 1)


def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


class ImageDataset(Dataset):

    def __init__(self, image_paths: List[str], transform: callable):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Do not crash the whole pipeline on a single bad file.
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image_path, image

    @classmethod
    def from_directory(cls, dataset_dir: str, transform: callable):
        image_paths = get_images_recursively(dataset_dir)
        return cls(image_paths, transform)

    @classmethod
    def get_file_size(file_path):
        return os.path.getsize(file_path)

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
                    selected_files = [
                        f for f in sorted_files if get_image_number(f) <= threshold
                    ]
                elif portion == "last":
                    selected_files = [
                        f
                        for f in sorted_files
                        if get_image_number(f) > 2 * threshold
                    ]
                else:
                    raise ValueError("portion must be either 'first' or 'last'")

                image_paths.extend(selected_files)

        return cls(image_paths, transform)


class DuplicateRemover(object):

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        threshold: float = 0.96,
        max_compare_size: int = 100000,
        dataloader_batch_size: int = 12,
        dataloader_num_workers: int = 6,
        pin_memory: bool = True,
        logger: Optional[logging.Logger] = None,
        # --- new (safe defaults; do not break external calls) ---
        max_iterations: int = 3,
        k_neighbors: Optional[int] = None,              # if None -> adaptive 300..500
        faiss_use_gpu: bool = False,                    # similarity search does NOT need GPU by default
        sharpness_keep_ratio: float = 0.90,             # keep those >= ratio * best_sharpness, then PNG tie-break
        score_image_max_side: int = 512,                # downscale for scoring (fast)
        score_num_workers: Optional[int] = None,        # threads for scoring PNG/sharpness
        rm_parallelism: int = 4,                        # xargs -P for rm
        use_amp: bool = True,                           # AMP for embedding speed
    ):

        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.threshold = float(threshold)
        self.max_compare_size = int(max_compare_size)  # kept for compatibility (not used in FAISS path)
        self.dataloader_batch_size = int(dataloader_batch_size)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.pin_memory = bool(pin_memory)

        self.max_iterations = int(max_iterations)
        self.k_neighbors = k_neighbors
        self.faiss_use_gpu = bool(faiss_use_gpu)
        self.sharpness_keep_ratio = float(sharpness_keep_ratio)
        self.score_image_max_side = int(score_image_max_side)
        self.rm_parallelism = int(rm_parallelism)
        self.use_amp = bool(use_amp)

        # scoring workers: default = physical cores (Linux), not SMT threads
        if score_num_workers is None:
            self.score_num_workers = get_physical_core_count(self.logger)
        else:
            self.score_num_workers = int(score_num_workers)

        # ---- model / transform ----
        self.logger.info(f"Loading {model_name} ...")

        # Prefer embedding head (num_classes=0) if supported by timm model
        try:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        except Exception:
            # fallback to original behavior
            self.model = timm.create_model(model_name, pretrained=True)

        # Multi-GPU embeddings: DataParallel
        self._num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if str(self.device).startswith("cuda") and self._num_gpus > 1 and (device is None or device == "cuda"):
            self.logger.info(f"Using DataParallel for embeddings across {self._num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.data_cfg = timm.data.resolve_data_config(
            self.model.module.pretrained_cfg if isinstance(self.model, nn.DataParallel) else self.model.pretrained_cfg
        )
        self.transform = timm.data.create_transform(**self.data_cfg)

        # ---- hybrid scoring cache (path -> (res_px, sharpness, png_size)) ----
        self._score_cache: Dict[str, Tuple[int, float, int]] = {}
        self._score_cache_lock = threading.Lock()

        # Laplacian kernel in PIL (fast)
        self._laplacian_filter = ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 1, 0,
                    1, -4, 1,
                    0, 1, 0],
            scale=1,
            offset=0
        )

        # Keep FAISS GPU resources alive if enabled (single-GPU path)
        self._faiss_res = None
        if FAISS_AVAILABLE and self.faiss_use_gpu and torch.cuda.is_available():
            try:
                self._faiss_res = faiss.StandardGpuResources()
            except Exception as e:
                self.logger.warning(f"Could not init FAISS GPU resources, falling back to CPU FAISS: {e}")
                self._faiss_res = None
                self.faiss_use_gpu = False

        if not FAISS_AVAILABLE:
            self.logger.warning(
                "FAISS not available. Similarity search fallback is much slower. "
                "Consider installing faiss-cpu / faiss-gpu."
            )

        # ---- FAISS CPU threads: use physical cores (exactness unchanged) ----
        self._faiss_cpu_threads = get_physical_core_count(self.logger)
        if FAISS_AVAILABLE:
            try:
                faiss.omp_set_num_threads(self._faiss_cpu_threads)
                self.logger.info(f"FAISS CPU threads set to {self._faiss_cpu_threads} (physical cores)")
            except Exception as e:
                self.logger.warning(f"Could not set FAISS CPU threads: {e}")

        # Keep dataset reference like the original code did (some pipelines expect this attr)
        self.dataset = None

    # ---------------- embeddings ----------------

    def compute_embeddings(self, dataset: ImageDataset) -> Tuple[List[str], np.ndarray]:
        dataloader = DataLoader(
            dataset,
            batch_size=self.dataloader_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory and str(self.device).startswith("cuda"),
            shuffle=False,
            persistent_workers=self.dataloader_num_workers > 0,
            prefetch_factor=2 if self.dataloader_num_workers > 0 else None,
        )

        all_paths: List[str] = []
        chunks: List[np.ndarray] = []

        use_cuda = str(self.device).startswith("cuda")
        amp_enabled = bool(self.use_amp and use_cuda)
        amp_device_type = "cuda" if use_cuda else "cpu"

        with torch.no_grad(), torch.autocast(device_type=amp_device_type, enabled=amp_enabled):
            for batch_paths, images in tqdm(dataloader, desc="Computing embeddings"):
                images = images.to(self.device, non_blocking=True)
                feats = self.model(images)

                # Make robust across different timm model outputs
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                if isinstance(feats, dict):
                    feats = feats.get("features", next(iter(feats.values())))

                if hasattr(feats, "ndim") and feats.ndim == 4:
                    feats = feats.mean(dim=(-2, -1))  # global average pool

                chunks.append(feats.detach().cpu().float().numpy())
                all_paths.extend(list(batch_paths))

        emb = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float32)
        return all_paths, emb

    # ---------------- hybrid scoring ----------------

    def _score_image_uncached(self, path: str) -> Tuple[int, float, int]:
        """
        Hybrid metrics (CPU):
          - res_px: width*height (from original image)
          - sharpness: Laplacian variance on downscaled grayscale
          - png_size: in-memory PNG (compress_level=1) size on downscaled RGB
        """
        try:
            im = Image.open(path)
            w, h = im.size
            res_px = int(w) * int(h)

            # Downscale for scoring (much faster)
            max_side = max(w, h)
            if max_side > self.score_image_max_side:
                scale = self.score_image_max_side / float(max_side)
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                im = im.resize((nw, nh), resample=Image.BICUBIC)

            im_rgb = im.convert("RGB")
            im_gray = im_rgb.convert("L")

            # Sharpness via Laplacian variance
            lap = im_gray.filter(self._laplacian_filter)
            arr = np.asarray(lap, dtype=np.float32)
            sharpness = float(arr.var())

            # PNG size (compress=1) in memory
            buf = io.BytesIO()
            im_rgb.save(buf, format="PNG", compress_level=1, optimize=False)
            png_size = int(buf.tell())

            return res_px, sharpness, png_size
        except Exception:
            return 0, 0.0, 0

    def _ensure_scores(self, paths: List[str]) -> None:
        missing = []
        with self._score_cache_lock:
            for p in paths:
                if p not in self._score_cache:
                    missing.append(p)

        if not missing:
            return

        results: Dict[str, Tuple[int, float, int]] = {}

        # Do not create more threads than tasks
        max_workers = min(self.score_num_workers, len(missing))
        if max_workers <= 1:
            for p in missing:
                results[p] = self._score_image_uncached(p)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for p, score in zip(missing, ex.map(self._score_image_uncached, missing)):
                    results[p] = score

        with self._score_cache_lock:
            self._score_cache.update(results)

    def _choose_best_in_component(self, member_ids: List[int], paths: List[str]) -> int:
        """
        Selection inside a duplicate-component:
          1) Prefer max resolution (w*h)
          2) Among those, prefer max sharpness (Laplacian var)
          3) Among sufficiently sharp (>= ratio * best), prefer max PNG-size(compress=1)
          4) Deterministic tie-breaker by path
        Returns: member_id (local index) to KEEP
        """
        comp_paths = [paths[i] for i in member_ids]
        self._ensure_scores(comp_paths)

        res: Dict[int, int] = {}
        sharp: Dict[int, float] = {}
        png: Dict[int, int] = {}

        with self._score_cache_lock:
            for mid in member_ids:
                rp, sh, ps = self._score_cache.get(paths[mid], (0, 0.0, 0))
                res[mid] = rp
                sharp[mid] = sh
                png[mid] = ps

        # 1) resolution gate
        max_res = max(res.values())
        candidates = [i for i in member_ids if res[i] == max_res]
        if len(candidates) == 1:
            return candidates[0]

        # 2) sharpness gate
        best_sharp = max(sharp[i] for i in candidates)
        if best_sharp <= 0:
            # fallback purely to png size, then path
            return max(candidates, key=lambda i: (png[i], paths[i]))

        keep_sharp = best_sharp * self.sharpness_keep_ratio
        sharp_candidates = [i for i in candidates if sharp[i] >= keep_sharp]
        if len(sharp_candidates) == 1:
            return sharp_candidates[0]

        # 3) png-size tie-break among sufficiently sharp
        return max(sharp_candidates, key=lambda i: (png[i], sharp[i], paths[i]))

    # ---------------- similarity / duplicates ----------------

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-8)

    class _UnionFind:
        def __init__(self, n: int):
            self.parent = np.arange(n, dtype=np.int32)
            self.rank = np.zeros(n, dtype=np.int8)

        def find(self, x: int) -> int:
            parent = self.parent
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(self, a: int, b: int) -> None:
            ra = self.find(a)
            rb = self.find(b)
            if ra == rb:
                return
            rank = self.rank
            parent = self.parent
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

    def _adaptive_k(self, n: int) -> int:
        if n <= 1:
            return 1
        if self.k_neighbors is not None:
            return max(2, min(int(self.k_neighbors), n))
        # defaults tuned for anime frames:
        # - minimum 300
        # - cap at 500 to limit work
        # - for small n -> k=n
        return min(n, max(300, min(500, n // 100 if n >= 10000 else 300)))

    def _find_duplicates_single_pass_faiss(
        self,
        embeddings: np.ndarray,
        paths: List[str],
    ) -> Tuple[Set[int], Set[int]]:
        """
        Build components via kNN edges (cosine similarity > threshold),
        then select 1 best per component using hybrid scoring.
        Returns (local_indices_to_remove, local_indices_to_keep).
        """
        n = len(embeddings)
        if n <= 1:
            return set(), set(range(n))

        # Ensure CPU FAISS uses all physical cores (exactness unchanged)
        if FAISS_AVAILABLE and (not self.faiss_use_gpu or self._faiss_res is None):
            try:
                faiss.omp_set_num_threads(self._faiss_cpu_threads)
            except Exception:
                pass

        k = self._adaptive_k(n)

        emb = self._l2_normalize(embeddings)
        d = emb.shape[1]

        index = faiss.IndexFlatIP(d)

        # Keep similarity search on CPU by default
        if self.faiss_use_gpu and self._faiss_res is not None:
            index = faiss.index_cpu_to_gpu(self._faiss_res, 0, index)

        index.add(emb)

        uf = self._UnionFind(n)

        # Stream union edges without storing full (n,k) matrices
        batch_size = 10000
        for start in tqdm(range(0, n, batch_size), desc=f"FAISS kNN (k={k})"):
            end = min(start + batch_size, n)
            sims, neigh = index.search(emb[start:end], k)

            for i in range(end - start):
                src = start + i
                row_sims = sims[i]
                row_neigh = neigh[i]
                for j in range(len(row_neigh)):
                    dst = int(row_neigh[j])
                    if dst < 0 or dst == src:
                        continue
                    if float(row_sims[j]) > self.threshold:
                        uf.union(src, dst)

        # Group members by component root
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = uf.find(i)
            groups.setdefault(r, []).append(i)

        to_remove: Set[int] = set()
        to_keep: Set[int] = set()

        for members in groups.values():
            if len(members) <= 1:
                to_keep.add(members[0])
                continue
            keep = self._choose_best_in_component(members, paths)
            to_keep.add(keep)
            for m in members:
                if m != keep:
                    to_remove.add(m)

        return to_remove, to_keep

    def _find_duplicates_single_pass_numpy(
        self,
        embeddings: np.ndarray,
        paths: List[str],
        chunk_size: int = 5000,
    ) -> Tuple[Set[int], Set[int]]:
        """
        Slow fallback if FAISS is missing.
        Still O(n^2) worst-case; intended only as fallback.
        Returns (local_indices_to_remove, local_indices_to_keep).
        """
        n = len(embeddings)
        if n <= 1:
            return set(), set(range(n))

        emb = self._l2_normalize(embeddings)
        uf = self._UnionFind(n)

        for start in tqdm(range(0, n, chunk_size), desc="Numpy similarity chunks"):
            end = min(start + chunk_size, n)
            sims = emb[start:end] @ emb.T  # (chunk, n)
            for i in range(end - start):
                src = start + i
                row = sims[i]
                row[src] = 0.0
                dsts = np.where(row > self.threshold)[0]
                for dst in dsts.tolist():
                    uf.union(src, int(dst))

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = uf.find(i)
            groups.setdefault(r, []).append(i)

        to_remove: Set[int] = set()
        to_keep: Set[int] = set()

        for members in groups.values():
            if len(members) <= 1:
                to_keep.add(members[0])
                continue
            keep = self._choose_best_in_component(members, paths)
            to_keep.add(keep)
            for m in members:
                if m != keep:
                    to_remove.add(m)

        return to_remove, to_keep

    def find_duplicates_iterative(self, embeddings: np.ndarray, paths: List[str]) -> Set[int]:
        """
        Iterative passes handle cases where kNN graph misses some edges for huge near-identical clusters.
        Returns global indices to remove.
        """
        all_to_remove: Set[int] = set()
        remaining = np.arange(len(embeddings), dtype=np.int32)

        for it in range(self.max_iterations):
            if len(remaining) <= 1:
                break

            cur_emb = embeddings[remaining]
            cur_paths = [paths[int(i)] for i in remaining]

            self.logger.info(f"Duplicate pass {it + 1}/{self.max_iterations}: {len(remaining)} images")

            if FAISS_AVAILABLE:
                local_remove, _local_keep = self._find_duplicates_single_pass_faiss(cur_emb, cur_paths)
            else:
                local_remove, _local_keep = self._find_duplicates_single_pass_numpy(cur_emb, cur_paths)

            if not local_remove:
                self.logger.info("No more duplicates found")
                break

            global_remove = {int(remaining[int(i)]) for i in local_remove}
            all_to_remove.update(global_remove)

            mask = np.ones(len(remaining), dtype=bool)
            for i in local_remove:
                mask[int(i)] = False
            remaining = remaining[mask]

            self.logger.info(f"Removed in pass: {len(local_remove)}; remaining: {len(remaining)}")

        return all_to_remove

    # --- compatibility method (kept from original API shape) ---
    def get_duplicate(
        self, embeddings: np.ndarray, indices: Optional[np.ndarray] = None
    ) -> Tuple[Set[int], Set[int]]:
        """
        Compatibility wrapper:
        returns (samples_to_remove, samples_to_keep) as GLOBAL indices,
        using the same hybrid logic and FAISS (if available).
        """
        if indices is None:
            indices = np.arange(len(embeddings))

        idx_list = indices.tolist()
        sub_emb = embeddings[indices]

        # Prefer dataset order paths if available
        if self.dataset is not None and hasattr(self.dataset, "image_paths"):
            sub_paths = [self.dataset.image_paths[i] for i in idx_list]
        else:
            # last resort: dummy paths (keeps deterministic but scoring will be poor)
            sub_paths = [str(i) for i in idx_list]

        if FAISS_AVAILABLE:
            local_remove, local_keep = self._find_duplicates_single_pass_faiss(sub_emb, sub_paths)
        else:
            local_remove, local_keep = self._find_duplicates_single_pass_numpy(sub_emb, sub_paths)

        global_remove = {idx_list[i] for i in local_remove}
        global_keep = {idx_list[i] for i in local_keep}
        return global_remove, global_keep

    # ---------------- public API (pipeline) ----------------

    def remove_similar(self, dataset: ImageDataset):
        start_time = time.time()
        self.logger.info(f"Compute embeddings for {len(dataset)} images ...")
        paths, embeddings = self.compute_embeddings(dataset)
        self.logger.info(f"Embedding computation took: {time.time() - start_time:.2f} seconds")

        if len(paths) <= 1:
            return

        start_time = time.time()
        samples_to_remove = self.find_duplicates_iterative(embeddings, paths)
        self.logger.info(f"Duplicate identification took: {time.time() - start_time:.2f} seconds")

        if not samples_to_remove:
            return

        files_to_remove = [paths[i] for i in samples_to_remove]

        start_time = time.time()
        if len(files_to_remove) > 200:
            # safe with spaces/newlines via -0, safe with leading dashes via rm -- ...
            payload = ("\0".join(files_to_remove) + "\0").encode()
            proc = subprocess.Popen(
                ["xargs", "-0", "-P", str(self.rm_parallelism), "rm", "-f", "--"],
                stdin=subprocess.PIPE,
            )
            proc.communicate(input=payload)
        else:
            for f in files_to_remove:
                try:
                    os.remove(f)
                except OSError:
                    pass
        self.logger.info(f"File removal took: {time.time() - start_time:.2f} seconds")

    def remove_similar_from_dir(self, dirpath: str, portion: Optional[str] = None):

        if portion:
            rtype = "op" if portion == "first" else "ed"
            self.logger.info(f"Removing {rtype} duplicates for '{dirpath}' ...")
            dataset = ImageDataset.from_subdirectories(
                dataset_dir=dirpath, transform=self.transform, portion=portion
            )
        else:
            self.logger.info(f"Removing duplicates for '{dirpath}' ...")
            dataset = ImageDataset.from_directory(dirpath, self.transform)

        if len(dataset) == 0:
            return

        # keep reference like original pipeline might expect
        self.dataset = dataset
        self.remove_similar(dataset)

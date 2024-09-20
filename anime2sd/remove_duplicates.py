import os
import logging
from tqdm import tqdm
from typing import List, Tuple, Set, Optional
from PIL import Image
from pathlib import Path
import subprocess
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import timm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .basics import get_related_paths, get_images_recursively

@staticmethod
def get_file_size(file_path):
        return os.path.getsize(file_path)

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
    ):

        if logger is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.logger.info(f"Loading {model_name} ...")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.max_compare_size = max_compare_size
        self.dataloader_batch_size = dataloader_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory
        self.data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**self.data_cfg)
        
    def compute_embeddings(self, dataset: ImageDataset):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            self.logger.warning("No GPU found. Falling back to CPU.")
            num_gpus = 1
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        subset_size = len(dataset) // num_gpus
        dataset_subsets = [
            torch.utils.data.Subset(dataset, range(i * subset_size, (i + 1) * subset_size))
            for i in range(num_gpus)
        ]
        dataset_subsets[-1] = torch.utils.data.Subset(
            dataset, range((num_gpus - 1) * subset_size, len(dataset))
        )

        dataloaders = [
            DataLoader(
                subset,
                batch_size=self.dataloader_batch_size,
                num_workers=self.dataloader_num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
            for subset in dataset_subsets
        ]

        embeddings = []

        for i, dataloader in enumerate(dataloaders):
            model = self.model.to(device)
            
            with torch.no_grad(), torch.autocast(device_type=device.type):
                for _, images in tqdm(dataloader, desc=f"GPU {i}"):
                    images = images.to(device)
                    features = model(images)
                    embeddings.append(features.cpu().float().numpy())

        return np.vstack(embeddings)

    def get_duplicate(
        self, embeddings: np.ndarray, indices: Optional[np.ndarray] = None
    ) -> Tuple[Set[int], Set[int]]:

        if indices is None:
            indices = np.arange(len(embeddings))
        embeddings = embeddings[indices]
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = similarity_matrix - np.identity(len(similarity_matrix))

        samples_to_remove = set()
        samples_to_keep = set()
        
        file_sizes = {idx: get_file_size(self.dataset.image_paths[idx]) for idx in indices}

        for idx in tqdm(range(len(embeddings))):
            sample_id = indices[idx]
            if sample_id not in samples_to_remove and sample_id not in samples_to_keep:
                dup_idxs = np.where(similarity_matrix[idx] > self.threshold)[0]
                if len(dup_idxs) > 0:
                    similar_images = [sample_id] + [indices[dup] for dup in dup_idxs]
                    largest_image = max(similar_images, key=lambda x: file_sizes[x])
                    samples_to_keep.add(largest_image)
                    samples_to_remove.update(set(similar_images) - {largest_image})
                else:
                    samples_to_keep.add(sample_id)

        return samples_to_remove, samples_to_keep

    def remove_similar(self, dataset: ImageDataset):

        start_time = time.time()
        self.logger.info(f"Compute embeddings for {len(dataset)} images ...")
        embeddings = self.compute_embeddings(dataset)
        embedding_time = time.time() - start_time
        self.logger.info(f"Embedding computation took: {embedding_time:.2f} seconds")

        start_time = time.time()
        samples_to_remove = set()

        for k in range(0, len(embeddings), self.max_compare_size):
            end = min(k + self.max_compare_size, len(embeddings))
            samples_to_remove_sub, _ = self.get_duplicate(
                embeddings, indices=np.arange(k, end)
            )
            samples_to_remove = samples_to_remove | samples_to_remove_sub
        duplicate_time = time.time() - start_time
        self.logger.info(f"Duplicate identification took: {duplicate_time:.2f} seconds")

        start_time = time.time()
        start_time = time.time()
        files_to_remove = [dataset.image_paths[sample_id] for sample_id in samples_to_remove] 
        file_list_time = time.time() - start_time
        self.logger.info(f"File list preparation took: {file_list_time:.2f} seconds")

        if files_to_remove:
            start_time = time.time()
            command = ["rm"] + files_to_remove
            subprocess.run(command, check=True)
            removal_time = time.time() - start_time
            self.logger.info(f"File removal took: {removal_time:.2f} seconds")

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
        self.dataset = dataset
        self.remove_similar(dataset)

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class ReplayTensorDataset(Dataset):
    """Replay dataset that mirrors CIFAR label type (Python int) for safe collation."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx].item())


class ReplayBuffer:
    """Simple class-aware replay buffer storing tensors on CPU."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = max_size
        self._rng = random.Random(seed)
        self._storage: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def __len__(self) -> int:
        return sum(len(v) for v in self._storage.values())

    def is_empty(self) -> bool:
        return len(self) == 0

    def add_samples(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        for img, lbl in zip(images, labels):
            self._storage[int(lbl.item())].append(img)
        self._trim_balanced()

    def _trim_balanced(self) -> None:
        total = len(self)
        if total <= self.max_size:
            return

        classes = [c for c, imgs in self._storage.items() if len(imgs) > 0]
        if not classes:
            return

        quota = max(1, self.max_size // len(classes))
        new_storage: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for cls in classes:
            imgs = self._storage[cls]
            if len(imgs) <= quota:
                new_storage[cls] = imgs
            else:
                idx = list(range(len(imgs)))
                self._rng.shuffle(idx)
                new_storage[cls] = [imgs[i] for i in idx[:quota]]

        # Fill leftover budget from remaining samples.
        used = sum(len(v) for v in new_storage.values())
        leftover = self.max_size - used
        if leftover > 0:
            pool: List[Tuple[int, torch.Tensor]] = []
            for cls in classes:
                chosen_ids = {id(x) for x in new_storage[cls]}
                for img in self._storage[cls]:
                    if id(img) not in chosen_ids:
                        pool.append((cls, img))
            self._rng.shuffle(pool)
            for cls, img in pool[:leftover]:
                new_storage[cls].append(img)

        self._storage = new_storage

    def sample_dataset(self) -> Optional[Dataset]:
        if self.is_empty():
            return None

        images = []
        labels = []
        for cls, imgs in self._storage.items():
            for img in imgs:
                images.append(img)
                labels.append(cls)

        idx = list(range(len(images)))
        self._rng.shuffle(idx)
        image_tensor = torch.stack([images[i] for i in idx], dim=0)
        label_tensor = torch.tensor([labels[i] for i in idx], dtype=torch.long)
        return ReplayTensorDataset(image_tensor, label_tensor)


class SplitCIFAR10Manager:
    """Split CIFAR-10 (5 tasks x 2 classes) with replay support."""

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int = 2,
        replay_size: int = 500,
        tasks: int = 5,
        classes_per_task: int = 2,
        seed: int = 0,
        task_order: Optional[List[int]] = None,
    ) -> None:
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tasks = tasks
        self.classes_per_task = classes_per_task
        self.seed = seed
        self.task_order = self._normalize_task_order(task_order)
        self.pin_memory = torch.cuda.is_available()

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        self.train_dataset = self._build_train_dataset(augment=True)
        self.train_dataset_eval = self._build_train_dataset(augment=False)
        self.test_dataset = self._build_test_dataset()

        self.task_classes = self._build_task_classes()
        self.train_indices_by_task = self._collect_task_indices(self.train_dataset)
        self.test_indices_by_task = self._collect_task_indices(self.test_dataset)

        self.replay_buffer = ReplayBuffer(max_size=replay_size, seed=seed)

    def _normalize_task_order(self, task_order: Optional[List[int]]) -> List[int]:
        if task_order is None:
            return list(range(self.tasks))

        if len(task_order) != self.tasks:
            raise ValueError(
                f"task_order must contain exactly {self.tasks} entries, got {len(task_order)}"
            )

        normalized = [int(v) for v in task_order]
        expected = set(range(self.tasks))
        if set(normalized) != expected:
            raise ValueError(
                f"task_order must be a permutation of {sorted(expected)}, got {normalized}"
            )
        return normalized

    def _build_train_dataset(self, augment: bool) -> Dataset:
        transform = self.train_transform if augment else self.test_transform
        try:
            return datasets.CIFAR10(
                root=self.root,
                train=True,
                transform=transform,
                download=True,
            )
        except Exception:
            # Offline fallback for smoke runs only.
            return datasets.FakeData(
                size=5000,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=transform,
            )

    def _build_test_dataset(self) -> Dataset:
        try:
            return datasets.CIFAR10(
                root=self.root,
                train=False,
                transform=self.test_transform,
                download=True,
            )
        except Exception:
            return datasets.FakeData(
                size=1000,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=self.test_transform,
            )

    def _dataset_targets(self, dataset: Dataset) -> List[int]:
        if hasattr(dataset, "targets"):
            return list(getattr(dataset, "targets"))
        labels = []
        for _, label in dataset:
            labels.append(int(label))
        return labels

    def _build_task_classes(self) -> List[List[int]]:
        base = [
            list(range(t * self.classes_per_task, (t + 1) * self.classes_per_task))
            for t in range(self.tasks)
        ]
        return [base[idx] for idx in self.task_order]

    def _collect_task_indices(self, dataset: Dataset) -> List[List[int]]:
        targets = self._dataset_targets(dataset)
        by_task: List[List[int]] = []
        for classes in self.task_classes:
            class_set = set(classes)
            by_task.append([i for i, y in enumerate(targets) if y in class_set])
        return by_task

    def _task_subset(self, dataset: Dataset, task_id: int) -> Dataset:
        if dataset is self.test_dataset:
            idxs = self.test_indices_by_task[task_id]
        elif dataset is self.train_dataset_eval:
            idxs = self.train_indices_by_task[task_id]
        else:
            idxs = self.train_indices_by_task[task_id]
        return Subset(dataset, idxs)

    def get_task_train_loader(self, task_id: int) -> DataLoader:
        current_subset = self._task_subset(self.train_dataset, task_id)
        replay_dataset = self.replay_buffer.sample_dataset()

        if replay_dataset is None:
            train_dataset: Dataset = current_subset
        else:
            train_dataset = ConcatDataset([current_subset, replay_dataset])

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def get_seen_train_loader(self, upto_task: int, batch_size: int) -> DataLoader:
        indices = []
        for t in range(upto_task + 1):
            indices.extend(self.train_indices_by_task[t])
        subset = Subset(self.train_dataset_eval, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def get_seen_test_loader(self, upto_task: int, batch_size: int) -> DataLoader:
        indices = []
        for t in range(upto_task + 1):
            indices.extend(self.test_indices_by_task[t])
        subset = Subset(self.test_dataset, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def get_task_test_loader(self, task_id: int, batch_size: Optional[int] = None) -> DataLoader:
        subset = self._task_subset(self.test_dataset, task_id)
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(
            subset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def update_replay_from_task(self, task_id: int, samples_per_class: Optional[int] = None) -> None:
        dataset = self.train_dataset_eval
        indices = self.train_indices_by_task[task_id]
        if samples_per_class is None:
            seen_classes = (task_id + 1) * self.classes_per_task
            samples_per_class = max(1, self.replay_buffer.max_size // max(1, seen_classes))

        class_to_indices: Dict[int, List[int]] = defaultdict(list)
        targets = self._dataset_targets(dataset)
        for idx in indices:
            class_to_indices[int(targets[idx])].append(idx)

        chosen_indices = []
        rng = random.Random(self.seed + task_id)
        for cls, cls_indices in class_to_indices.items():
            rng.shuffle(cls_indices)
            chosen_indices.extend(cls_indices[:samples_per_class])

        imgs = []
        lbls = []
        for idx in chosen_indices:
            img, lbl = dataset[idx]
            imgs.append(img)
            lbls.append(lbl)

        if imgs:
            self.replay_buffer.add_samples(
                torch.stack(imgs, dim=0),
                torch.tensor(lbls, dtype=torch.long),
            )

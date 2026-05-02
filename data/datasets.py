import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class ReplayTensorDataset(Dataset):
    """Replay dataset backed by source indices for on-the-fly augmentation."""

    def __init__(self, source_dataset: Dataset, indices: List[int], labels: List[int]) -> None:
        self.source_dataset = source_dataset
        self.indices = [int(idx) for idx in indices]
        self.labels = torch.tensor([int(lbl) for lbl in labels], dtype=torch.long)
        if len(self.indices) != int(self.labels.size(0)):
            raise ValueError("indices and labels must have the same length")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, _ = self.source_dataset[self.indices[idx]]
        return image, int(self.labels[idx].item())


class DatasetWithReplayFlag(Dataset):
    """Attach replay-source flag so training can gate distillation on replay only."""

    def __init__(self, dataset: Dataset, is_replay: bool) -> None:
        self.dataset = dataset
        self.is_replay = 1 if is_replay else 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, int(label), self.is_replay


class ReplayBuffer:
    """Simple class-aware replay buffer storing source dataset indices."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = max_size
        self._rng = random.Random(seed)
        self._storage: Dict[int, List[int]] = defaultdict(list)

    def __len__(self) -> int:
        return sum(len(v) for v in self._storage.values())

    def is_empty(self) -> bool:
        return len(self) == 0

    def add_indices(self, indices: List[int], labels: List[int]) -> None:
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have the same length")
        for sample_idx, lbl in zip(indices, labels):
            self._storage[int(lbl)].append(int(sample_idx))
        self._trim_balanced()

    def replace(self, class_to_indices: Dict[int, List[int]]) -> None:
        new_storage: Dict[int, List[int]] = defaultdict(list)
        for cls, indices in class_to_indices.items():
            if not indices:
                continue
            new_storage[int(cls)] = [int(idx) for idx in indices]
        self._storage = new_storage

    def _trim_balanced(self) -> None:
        total = len(self)
        if total <= self.max_size:
            return

        classes = [c for c, samples in self._storage.items() if len(samples) > 0]
        if not classes:
            return

        quota = self.max_size // len(classes)
        if quota <= 0:
            self._storage = defaultdict(list)
            return
        new_storage: Dict[int, List[int]] = defaultdict(list)

        for cls in classes:
            samples = self._storage[cls]
            if len(samples) <= quota:
                new_storage[cls] = list(samples)
            else:
                idx = list(range(len(samples)))
                self._rng.shuffle(idx)
                new_storage[cls] = [samples[i] for i in idx[:quota]]

        self._storage = new_storage

    def sample_dataset(self, source_dataset: Dataset) -> Optional[Dataset]:
        if self.is_empty():
            return None

        indices = []
        labels = []
        for cls, class_indices in self._storage.items():
            for sample_idx in class_indices:
                indices.append(sample_idx)
                labels.append(cls)

        idx = list(range(len(indices)))
        self._rng.shuffle(idx)
        shuffled_indices = [indices[i] for i in idx]
        shuffled_labels = [labels[i] for i in idx]
        return ReplayTensorDataset(source_dataset, shuffled_indices, shuffled_labels)


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
        current_dataset: Dataset = DatasetWithReplayFlag(current_subset, is_replay=False)
        replay_dataset = self.replay_buffer.sample_dataset(self.train_dataset)

        if replay_dataset is None:
            train_dataset: Dataset = current_dataset
        else:
            train_dataset = ConcatDataset(
                [
                    current_dataset,
                    DatasetWithReplayFlag(replay_dataset, is_replay=True),
                ]
            )

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
        seen_task_ids = list(range(task_id + 1))
        seen_classes = sorted({int(cls) for t in seen_task_ids for cls in self.task_classes[t]})
        if not seen_classes:
            self.replay_buffer.replace({})
            return

        if samples_per_class is None:
            samples_per_class = self.replay_buffer.max_size // len(seen_classes)
        samples_per_class = int(samples_per_class)
        if samples_per_class <= 0:
            self.replay_buffer.replace({})
            return

        # Strict CL: only use current-task data directly; past data must come from replay buffer.
        targets = self._dataset_targets(self.train_dataset)
        class_to_pool: Dict[int, List[int]] = {cls: [] for cls in seen_classes}
        current_task_classes = {int(cls) for cls in self.task_classes[task_id]}
        for idx in self.train_indices_by_task[task_id]:
            cls = int(targets[idx])
            if cls in class_to_pool:
                class_to_pool[cls].append(int(idx))

        for cls, stored_indices in self.replay_buffer._storage.items():
            cls_int = int(cls)
            if cls_int in class_to_pool and cls_int not in current_task_classes:
                class_to_pool[cls_int].extend(int(i) for i in stored_indices)

        rng = random.Random(self.seed + 10_000 + task_id)
        balanced_selection: Dict[int, List[int]] = {}
        for cls in seen_classes:
            pool = class_to_pool.get(cls, [])
            if not pool:
                continue
            candidates = list(pool)
            rng.shuffle(candidates)
            if len(candidates) >= samples_per_class:
                balanced_selection[cls] = candidates[:samples_per_class]
            else:
                # Keep strict class counts even when a class pool is undersized.
                extra = [rng.choice(candidates) for _ in range(samples_per_class - len(candidates))]
                balanced_selection[cls] = candidates + extra

        self.replay_buffer.replace(balanced_selection)

"""
Replay Buffer implementations for Continual Learning.

This module provides buffer implementations for storing and replaying samples
in continual learning scenarios, supporting reservoir sampling and class-balanced strategies.
"""

import numpy as np
import random
from collections import defaultdict
from lightning.fabric.utilities import move_data_to_device


class SimpleBuffer():
    """
    Simple replay buffer using reservoir sampling.

    This buffer stores samples without considering class balance,
    using reservoir sampling to maintain a fixed-size buffer.
    """
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffer = []  # List of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples = 0
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for i in range(len(self.buffer)):
                self.buffer[i] = move_data_to_device(self.buffer[i], device)
        return self

    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        # In this simple version, we do not adjust the buffer size
        pass

    def add_judge(self, sample_label):
        """Determine whether to add a new sample to the buffer using reservoir sampling."""
        n_seen = self.n_seen_samples
        self.n_seen_samples += 1
        if len(self.buffer) < self.buffer_size:
            # Buffer not full yet
            return len(self.buffer)
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size:
                return rand_idx
            else:
                return -1  # Do not add the sample

    def add(self, sample, idx):
        """Add a sample to the buffer at the given index."""
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffer):
            # Adding a new sample to the buffer
            self.buffer.append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            # Replacing an existing sample in the buffer
            self.n_patches_total = self.n_patches_total - self.buffer[idx]['features'].size(0) + sample['features'].size(0)
            old_label = self.buffer[idx]['label'].item()
            self.labels[old_label] -= 1
            self.buffer[idx] = sample
            self.labels[label] = self.labels.get(label, 0) + 1

    def start_epoch(self):
        """Reset indices for a new epoch."""
        self.epoch_indices = np.random.permutation(len(self.buffer))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get the next batch of samples for the current epoch."""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.buffer) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.buffer[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch."""
        return self.samples_output_in_epoch

    def __len__(self):
        return len(self.buffer)


class ClsssIncrementalBuffer():
    """
    Class-balanced replay buffer for continual learning.

    This buffer maintains equal representation of each class by allocating
    buffer space equally across classes.
    """
    def __init__(self, buffer_size, device='cpu', **kwargs):
        self.buffers = {}  # Dict mapping class labels to lists of samples
        self.buffer_size = buffer_size
        self.device = device
        self.sample_selection_strategy = 'reservoir'
        self.n_seen_samples_per_class = defaultdict(int)
        self.num_classes = 0
        self.buffer_size_per_class = None
        self.epoch_indices = None
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0
        self.n_patches_total = 0
        self.labels = {}  # Count of samples per class

    def to(self, device):
        old_device = self.device
        self.device = device
        if old_device != device:
            for l in self.buffers:
                for i in range(len(self.buffers[l])):
                    self.buffers[l][i] = move_data_to_device(self.buffers[l][i], device)
        return self

    # Given the number of new classes will be adding, adjust the buffer size per class
    def adjust_buffer_size_by_new_classes(self, n_new_classes: int):
        self.buffer_size_per_class = self.buffer_size // (self.num_classes + n_new_classes)
        for l in self.buffers:
            if len(self.buffers[l]) > self.buffer_size_per_class:
                self.buffers[l] = random.sample(self.buffers[l], self.buffer_size_per_class)
                self.labels[l] = self.buffer_size_per_class
        # calculate the number of patches in the buffer
        self.n_patches_total = sum([sample['features'].size(0) for samples in self.buffers.values() for sample in samples])

    # Judge whether to add the new sample to the buffer by reservoir sampling
    # If add, return the index
    def add_judge(self, sample_label):
        if sample_label not in self.buffers:
            # New class encountered
            self.buffers[sample_label] = []
            self.num_classes += 1

        self.n_seen_samples_per_class[sample_label] += 1
        n_seen = self.n_seen_samples_per_class[sample_label]
        if len(self.buffers[sample_label]) < self.buffer_size_per_class:
            return len(self.buffers[sample_label])
        else:
            # Buffer full, apply reservoir sampling
            rand_idx = np.random.randint(0, n_seen)
            if rand_idx < self.buffer_size_per_class:
                return rand_idx
            else:
                return -1

    def add(self, sample, idx):
        assert idx != -1, "The sample is not added to the buffer"
        sample = move_data_to_device(sample, self.device)
        label = sample['label'].item()
        if idx == len(self.buffers[label]):
            self.buffers[label].append(sample)
            self.n_patches_total += sample['features'].size(0)
            self.labels[label] = self.labels.get(label, 0) + 1
        else:
            self.n_patches_total = self.n_patches_total - self.buffers[label][idx]['features'].size(0) + sample['features'].size(0)
            self.buffers[label][idx] = sample
            # Labels count remains the same

    def start_epoch(self):
        """Reset indices for a new epoch"""
        # Flatten buffers into a single list
        self.all_samples = []
        for samples in self.buffers.values():
            self.all_samples.extend(samples)
        self.epoch_indices = np.random.permutation(len(self.all_samples))
        self.current_epoch_position = 0
        self.samples_output_in_epoch = 0

    def get_next_batch(self, batch_size=1):
        """Get next batch of samples for the current epoch"""
        if self.epoch_indices is None:
            self.start_epoch()
        remaining = len(self.all_samples) - self.current_epoch_position
        if remaining == 0:
            return None  # Epoch is finished
        actual_batch_size = min(batch_size, remaining)
        batch_indices = self.epoch_indices[self.current_epoch_position:
                                           self.current_epoch_position + actual_batch_size]
        batch = [self.all_samples[i] for i in batch_indices]
        self.current_epoch_position += actual_batch_size
        self.samples_output_in_epoch += actual_batch_size
        return batch

    def get_samples_output_count(self):
        """Return the number of samples output in the current epoch"""
        return self.samples_output_in_epoch

    def __len__(self):
        return sum(len(samples) for samples in self.buffers.values())

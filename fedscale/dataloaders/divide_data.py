# -*- coding: utf-8 -*-
import csv
import logging
import random
import time
from collections import defaultdict
from random import Random

import numpy as np
from torch.utils.data import DataLoader

#from argParser import args


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, args, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.numOfLabels = numOfClass
        self.client_label_cnt = defaultdict(set)

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def getClientLen(self):
        return len(self.partitions)

    def getClientLabel(self):
        return [len(self.client_label_cnt[i]) for i in range(self.getClientLen())]

    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file and align it to the dataset indices.

        Format assumption: CSV columns contain at least
            0: client_id
            1: sample_path (relative name used by the dataset)
            -1: label_id (optional)

        We map sample_path to the actual index in ``self.data`` so that
        partition indices are guaranteed to be valid for the underlying
        dataset. This avoids IndexError when dataset content or ordering
        differs from the CSV enumeration.
        """
        logging.info(f"Partitioning data by profile {data_map_file}...")

        # Build a name->index map from the instantiated dataset when possible.
        # Some datasets (e.g., Google Speech) expose file names via
        # ``self.data.data``. Others (e.g., StackOverflow NLP) expose token
        # sequences (lists), which are unhashable and cannot serve as keys.
        name_to_idx = {}
        try:
            dataset_names = list(getattr(self.data, 'data'))
            # Only construct the map if entries are hashable (e.g., str paths)
            if len(dataset_names) > 0:
                try:
                    _ = hash(dataset_names[0])
                    name_to_idx = {name: idx for idx, name in enumerate(dataset_names)}
                except TypeError:
                    # Unhashable entries (e.g., list tokens) – skip name mapping
                    name_to_idx = {}
        except Exception:
            name_to_idx = {}

        client_dict: dict[int, list[int]] = {}
        total_rows = 0
        mapped_rows = 0
        missing_rows = 0

        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader, None)  # skip header if present
            if header:
                logging.info(f"Trace names are {', '.join(header)}")

            for row in csv_reader:
                total_rows += 1
                try:
                    client_id = int(row[0])
                except Exception:
                    # Malformed row – skip
                    missing_rows += 1
                    continue

                sample_name = row[1] if len(row) > 1 else None
                if sample_name is None:
                    missing_rows += 1
                    continue
                # Two possibilities for the mapping's second column:
                # 1) A numeric dataset index (used by StackOverflow TextDataset)
                # 2) A file/sample name that we need to resolve via name_to_idx
                idx = None
                # Try numeric index first
                try:
                    cand = int(sample_name)
                    if 0 <= cand < len(self.data):
                        idx = cand
                except Exception:
                    pass
                # Fall back to name lookup if available
                if idx is None and name_to_idx:
                    idx = name_to_idx.get(sample_name, None)
                if idx is None:
                    # The CSV lists a sample not present in the local dataset.
                    # Skip it to keep indices valid.
                    missing_rows += 1
                    continue

                client_dict.setdefault(client_id, []).append(idx)
                self.client_label_cnt[client_id].add(row[-1] if row else None)
                mapped_rows += 1

        # Expose the mapping expected by the runtime
        self.client_dict = client_dict
        sorted_ids = sorted(client_dict.keys())
        self.partitions = [client_dict[cid] for cid in sorted_ids]

        if missing_rows:
            logging.warning(
                "trace_partition: %d/%d samples from mapping were not found in the dataset and were skipped",
                missing_rows, total_rows,
            )

    def partition_data_helper(self, num_clients, data_map_file=None):

        # read mapping file to partition trace
        if data_map_file is not None:
            self.trace_partition(data_map_file)
        else:
            self.uniform_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        # random partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(f"Randomly partitioning data, {data_len} samples...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition, istest):
        resultIndex = self.partitions[partition % len(self.partitions)]

        exeuteLength = len(resultIndex) if not istest else int(
            len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}
    
    def get_partition_by_client(self, client_id: int, istest: bool):
        """Return a torch-Compatible `Partition` object for the given
        *real* client id."""
        idx_list = list(self.client_dict.get(client_id, []))  # copy to avoid in-place shuffles
        if istest:
            idx_list = idx_list[: int(len(idx_list) * self.args.test_ratio)]
        self.rng.shuffle(idx_list)
        return Partition(self.data, idx_list)


def select_dataset(rank, partition, batch_size, args, isTest=False, collate_fn=None):
    """Load data given client Id"""
    # If the partition was created from a CSV we have one shard per
    # real-client id; otherwise fall back to the original modulo logic.
    if hasattr(partition, "client_dict"):
        partition = partition.get_partition_by_client(rank, isTest)
    else:
        partition = partition.use(rank - 1, isTest)
    dropLast = False if isTest else True
    if isTest:
        num_loaders = 0
    else:
        num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import collections
import csv
import gc
import glob
import logging
import os
import pickle
import random
import re
import shutil
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

"""Configure how many CPU workers the pre-processing pool uses.
Defaults to all cores, but can be capped via FEDSCALE_PREPROC_WORKERS to
reduce memory/IO pressure when multiple executors run on the same host.
"""
N_JOBS = min(cpu_count(), int(os.environ.get("FEDSCALE_PREPROC_WORKERS", str(cpu_count()))))
logger = logging.getLogger(__name__)


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def feature_creation_worker(files, tokenizer, block_size, worker_idx):
    examples = []
    sample_client = []
    client_mapping = collections.defaultdict(list)

    user_id = -1
    start_time = time.time()
    for idx, file in enumerate(files):
        try:
            with open(file, encoding="utf-8", errors='ignore') as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text))
            if len(tokenized_text) > 0:
                user_id += 1

            # Truncate in block of block_size
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                examples.append(tokenizer.build_inputs_with_special_tokens(
                    tokenized_text[i: i + block_size]))
                client_mapping[user_id].append(len(examples)-1)
                sample_client.append(user_id)
        except Exception:
            logging.exception(f"Worker {worker_idx}: exception while processing file {file}")
        if idx % 10000 == 0:
            logging.info(f"Task {worker_idx}: {len(files)-idx} files left, {idx} files complete, remaining time {(time.time()-start_time)/(idx+1)*(len(files)-idx)}")
            gc.collect()

    return (examples, client_mapping, sample_client)


def _feature_creation_worker_star(args):
    # Helper to allow imap_unordered with tuple args
    return feature_creation_worker(*args)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):

        block_size = block_size - \
            (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory = file_path
        cached_features_file = os.path.join(
            directory, args.model + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            gc.disable()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                self.client_mapping = pickle.load(handle)
            gc.enable()
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            self.sample_client = []
            self.client_mapping = collections.defaultdict(list)
            user_id = -1

            files = [entry.name for entry in os.scandir(
                file_path) if '_cached_lm_' not in entry.name]
            # make sure files are ordered
            files = [os.path.join(file_path, x) for x in sorted(files)]
            # Break the file list into many small tasks to reduce per-result payload
            files_per_task = int(os.environ.get("FEDSCALE_PREPROC_FILES_PER_TASK", "2000"))
            task_slices = []
            for i in range(0, len(files), max(1, files_per_task)):
                task_slices.append(files[i:i+files_per_task])
            logger.info("Preproc will use %d tasks (files_per_task=%d) over %d workers",
                        len(task_slices), files_per_task, N_JOBS)

            # Coordinate across processes/hosts so only one builds the cache
            lock_path = cached_features_file + ".lock"
            tmp_cache = cached_features_file + ".tmp"

            def _acquire_lock(path):
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    return True
                except FileExistsError:
                    return False

            if not args.overwrite_cache and not _acquire_lock(lock_path):
                logger.info("Another process is building cache %s, waiting ...", cached_features_file)
                start_wait = time.time()
                wait_timeout = int(os.environ.get("FEDSCALE_PREPROC_LOCK_TIMEOUT", "1800"))
                while not os.path.exists(cached_features_file):
                    time.sleep(5)
                    # If we have been waiting too long, treat lock as stale and rebuild
                    if time.time() - start_wait > wait_timeout:
                        try:
                            # If lock is stale, remove it
                            if os.path.exists(lock_path):
                                lock_age = time.time() - os.path.getmtime(lock_path)
                                if lock_age > wait_timeout:
                                    logger.warning("Stale lock detected for %s (age %.1fs). Breaking lock and rebuilding.", cached_features_file, lock_age)
                                    os.remove(lock_path)
                        except Exception:
                            pass
                        # Try to acquire lock again; if success, fall through to build path
                        if _acquire_lock(lock_path):
                            break
                if os.path.exists(cached_features_file):
                    logger.info("Detected cache ready at %s, loading ...", cached_features_file)
                    gc.disable()
                    with open(cached_features_file, "rb") as handle:
                        self.examples = pickle.load(handle)
                        self.client_mapping = pickle.load(handle)
                        try:
                            self.sample_client = pickle.load(handle)
                        except Exception:
                            pass
                    gc.enable()
                else:
                    logger.info("Proceeding to build cache for %s after wait/lock handling.", cached_features_file)
                    # Build cache; stream results to reduce peak memory
                    try:
                        pool_inputs = []
                        pool = Pool(N_JOBS)
                        worker_cnt = 0
                        for begin, end in chunks_idx(range(len(files)), N_JOBS):
                            pool_inputs.append([files[begin:end], tokenizer, block_size, worker_cnt])
                            worker_cnt += 1

                        user_id_base = 0
                        for (examples, client_mapping, sample_client) in pool.imap_unordered(_feature_creation_worker_star, pool_inputs, chunksize=1):
                            self.examples += examples
                            true_sample_client = [i + user_id_base for i in sample_client]
                            self.sample_client += true_sample_client
                            for user_id, true_user_id in zip(sample_client, true_sample_client):
                                self.client_mapping[true_user_id] = client_mapping[user_id]
                            if len(true_sample_client) > 0:
                                user_id_base = true_sample_client[-1] + 1
                        pool.close()
                        pool.join()
                    except Exception:
                        logging.exception("Pre-processing pool failed; falling back to single-process build")
                        examples, client_mapping, sample_client = feature_creation_worker(files, tokenizer, block_size, 0)
                        self.examples = examples
                        self.sample_client = sample_client
                        self.client_mapping = client_mapping

                    # Atomic write of cache
                    logger.info("Saving features into cached file %s", cached_features_file)
                    with open(tmp_cache, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=-1)
                        pickle.dump(self.client_mapping, handle, protocol=-1)
                        pickle.dump(self.sample_client, handle, protocol=-1)
                    os.replace(tmp_cache, cached_features_file)
                    if os.path.exists(lock_path):
                        try:
                            os.remove(lock_path)
                        except Exception:
                            pass
            else:
                # Build cache; stream results to reduce peak memory
                try:
                    # If only one worker is requested, avoid multiprocessing entirely.
                    if N_JOBS <= 1 or os.environ.get("FEDSCALE_PREPROC_FORCE_SEQUENTIAL", "0") == "1":
                        logger.info("Sequential preproc mode: %d tasks", len(task_slices))
                        user_id_base = 0
                        for task_idx, sl in enumerate(task_slices):
                            examples, client_mapping, sample_client = feature_creation_worker(sl, tokenizer, block_size, task_idx)
                            self.examples += examples
                            true_sample_client = [i + user_id_base for i in sample_client]
                            self.sample_client += true_sample_client
                            for user_id, true_user_id in zip(sample_client, true_sample_client):
                                self.client_mapping[true_user_id] = client_mapping[user_id]
                            if len(true_sample_client) > 0:
                                user_id_base = true_sample_client[-1] + 1
                    else:
                        pool_inputs = []
                        pool = Pool(N_JOBS)
                        worker_cnt = 0
                        for sl in task_slices:
                            pool_inputs.append([sl, tokenizer, block_size, worker_cnt])
                            worker_cnt += 1

                        user_id_base = 0
                        for (examples, client_mapping, sample_client) in pool.imap_unordered(_feature_creation_worker_star, pool_inputs, chunksize=1):
                            self.examples += examples
                            true_sample_client = [i + user_id_base for i in sample_client]
                            self.sample_client += true_sample_client
                            for user_id, true_user_id in zip(sample_client, true_sample_client):
                                # Note: client_mapping indexes are local to the chunk; they may not be used later
                                self.client_mapping[true_user_id] = client_mapping[user_id]
                            if len(true_sample_client) > 0:
                                user_id_base = true_sample_client[-1] + 1
                        pool.close()
                        pool.join()
                except Exception:
                    logging.exception("Pre-processing pool failed; falling back to single-process build")
                    examples, client_mapping, sample_client = feature_creation_worker(files, tokenizer, block_size, 0)
                    self.examples = examples
                    self.sample_client = sample_client
                    self.client_mapping = client_mapping

                # Atomic write of cache
                logger.info("Saving features into cached file %s", cached_features_file)
                with open(tmp_cache, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=-1)
                    pickle.dump(self.client_mapping, handle, protocol=-1)
                    pickle.dump(self.sample_client, handle, protocol=-1)
                os.replace(tmp_cache, cached_features_file)

                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except Exception:
                        pass

            # dump the data_mapping_file
            results = [['client_id', 'sample_path', 'label_name', 'label_id']]
            for i in range(len(self.sample_client)):
                results.append([self.sample_client[i], i, -1, -1])

            # Ensure the output directory exists to avoid FileNotFoundError
            out_dir = os.path.join(file_path, '../client_data_mapping')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'result.csv'), 'w') as csvFile:
                writer = csv.writer(csvFile)
                for line in results:
                    writer.writerow(line)

        self.data = self.examples
        self.targets = [0 for i in range(len(self.data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = os.path.join(args.data_dir, 'test') if evaluate else os.path.join(
        args.data_dir, 'train')

    return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def mask_tokens(inputs, tokenizer, args, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling.
    Operates on the same device as `inputs` to avoid device-mismatch errors.
    80% MASK, 10% random, 10% original.
    """
    dev = inputs.device
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    probability_matrix = torch.full(labels.shape, args.mlm_probability, device=dev)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=dev), value=0.0)
    # Honor padding token if defined by the tokenizer implementation
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is not None:
        padding_mask = labels.eq(pad_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).to(dtype=torch.bool, device=dev)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=dev)).to(dtype=torch.bool) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=dev)).to(dtype=torch.bool) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=dev)
    bool_indices_random = indices_random
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

from datasets.document_dataset_base import DocumentDatasetBase
from torch.utils.data.sampler import Sampler

import numpy as np
from random import shuffle


class DocumentSampler(Sampler):

    def __init__(
            self,
            dataset: DocumentDatasetBase,
            shuffle: bool = True,
            batch_size=64):

        self._total_length = len(dataset)
        self._shuffle = shuffle
        self._ids_per_doc = dataset.get_indices_per_document()
        self.batch_size = batch_size
        self.iter_list = self._create_iter_list()

    def _create_iter_list(self):
        data_buckets = {
            k: np.asarray(v) for k, v in self._ids_per_doc.items()
        }

        bucket_keys = list(data_buckets.keys())

        if self._shuffle:
            shuffle(bucket_keys)

        iter_list = []
        for k in bucket_keys:
            np.random.shuffle(data_buckets[k])
            if len(data_buckets[k]) < self.batch_size:
                iter_list.append(data_buckets[k])
                continue

            iter_list += (np.array_split(
                data_buckets[k],
                int(data_buckets[k].shape[0]/self.batch_size)))

        # shuffle all the batches so they are not ordered by bucket
        # size

        self._total_length = len(iter_list)
        return iter_list

    def __iter__(self):
        for i in self.iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return self._total_length
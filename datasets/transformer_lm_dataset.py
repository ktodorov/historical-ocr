from enums.ocr_output_type import OCROutputType
from enums.run_type import RunType
import os
import numpy as np
import torch
import pickle
from overrides import overrides

from datasets.dataset_base import DatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.file_service import FileService
from services.mask_service import MaskService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.log_service import LogService

from services.process.transformer_process_service import TransformerProcessService

class TransformerLMDataset(DatasetBase):
    def __init__(
            self,
            language: str,
            arguments_service: OCRQualityArgumentsService,
            process_service: TransformerProcessService,
            mask_service: MaskService,
            **kwargs):
        super().__init__()

        self._mask_service = mask_service
        self._arguments_service = arguments_service

        self._entries = process_service.get_entries(self._arguments_service.ocr_output_type)

    @overrides
    def __len__(self):
        return len(self._entries)

    @overrides
    def __getitem__(self, idx):
        entry = self._entries[idx]
        return (entry.token_ids, entry.mask_ids)

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        tokens_ids, masks = batch_split
        lengths = [len(sequence) for sequence in tokens_ids]
        max_length = max(lengths)

        padded_sequences = np.zeros((batch_size, max_length), dtype=np.int64)
        padded_masks = np.ones((batch_size, max_length), dtype=np.int64)

        for i, l in enumerate(lengths):
            padded_sequences[i][0:l] = tokens_ids[i][0:l]
            padded_masks[i][0:l] = masks[i][0:l]

        return self._sort_batch(
            torch.from_numpy(padded_sequences).to(
                self._arguments_service.device),
            torch.from_numpy(padded_masks).bool().to(
                self._arguments_service.device),
            torch.tensor(lengths).to(self._arguments_service.device))

    def _sort_batch(self, sequences, masks, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = sequences[perm_idx]
        mask_tensor = masks[perm_idx]
        return self._mask_service.mask_tokens(seq_tensor, mask_tensor, seq_lengths)

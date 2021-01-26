from typing import Dict, List
import numpy as np
import torch
from overrides import overrides

from datasets.document_dataset_base import DocumentDatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.mask_service import MaskService
from services.log_service import LogService

from services.process.transformer_process_service import TransformerProcessService

class TransformerLMDataset(DocumentDatasetBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: TransformerProcessService,
            mask_service: MaskService,
            log_service: LogService,
            **kwargs):
        super().__init__()

        self._mask_service = mask_service
        self._arguments_service = arguments_service
        self._log_service = log_service

        self._entries = process_service.get_entries(self._arguments_service.ocr_output_type)
        self._log_service.log_debug(f'Loaded {len(self._entries)} entries in transformer dataset')

    @overrides
    def __len__(self):
        return len(self._entries)

    @overrides
    def __getitem__(self, ids):
        entries = [self._entries[idx] for idx in ids]
        batch_size = len(ids)

        tokens_ids = [entry.token_ids for entry in entries]
        masks = [entry.mask_ids for entry in entries]

        lengths = [len(sequence) for sequence in tokens_ids]
        max_length = max(lengths)

        padded_sequences = np.zeros((batch_size, max_length), dtype=np.int64)
        if self._arguments_service.padding_idx != 0:
            padded_sequences.fill(self._arguments_service.padding_idx)

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

    @overrides
    def get_indices_per_document(self) -> Dict[int, List[int]]:
        total_documents = len(set([x.document_index for x in self._entries]))
        result = {
            i: []
            for i in range(total_documents)
        }

        for i, entry in enumerate(self._entries):
            result[entry.document_index].append(i)

        return result
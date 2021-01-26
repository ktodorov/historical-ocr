import numpy as np
from overrides import overrides
import torch

from datasets.dataset_base import DatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.process.evaluation_process_service import EvaluationProcessService
from services.log_service import LogService

class EvaluationDataset(DatasetBase):
    def __init__(
        self,
        arguments_service: OCRQualityArgumentsService,
        process_service: EvaluationProcessService,
        log_service: LogService):
        self._arguments_service = arguments_service
        self._process_service = process_service
        self._log_service = log_service

        self._target_tokens = self._process_service.get_target_tokens()
        self._log_service.log_debug(f'Loaded {len(self._target_tokens)} target tokens in evaluation dataset')

    @overrides
    def __len__(self):
        return len(self._target_tokens)

    @overrides
    def __getitem__(self, idx):
        target_token = self._target_tokens[idx]
        return target_token.token, target_token.vocabulary_ids

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        tokens, vocab_ids = batch_split

        all_padded_sequences = []

        for n in range(len(vocab_ids[0])):
            current_vocab_ids = [x[n] for x in vocab_ids]
            lengths = [len(sequence) for sequence in current_vocab_ids]
            max_length = max(lengths)
            padded_sequences = np.zeros((batch_size, max_length), dtype=np.int64)
            if self._arguments_service.padding_idx != 0:
                padded_sequences.fill(self._arguments_service.padding_idx)

            for i, l in enumerate(lengths):
                padded_sequences[i][0:l] = current_vocab_ids[i][0:l]

            all_padded_sequences.append(torch.Tensor(padded_sequences).long().to(self._arguments_service.device))

        return tokens, all_padded_sequences
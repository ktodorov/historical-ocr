import numpy as np
from overrides import overrides
import torch

from datasets.dataset_base import DatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.process.evaluation_process_service import EvaluationProcessService

class EvaluationDataset(DatasetBase):
    def __init__(
        self,
        arguments_service: OCRQualityArgumentsService,
        process_service: EvaluationProcessService):
        self._arguments_service = arguments_service
        self._process_service = process_service

        self._target_tokens = self._process_service.get_target_tokens()

    @overrides
    def __len__(self):
        return len(self._target_tokens)

    @overrides
    def __getitem__(self, idx):
        target_token = self._target_tokens[idx]
        return (target_token.word, target_token.vocabulary_id)

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_split = list(zip(*DataLoaderBatch))

        words, token_ids = batch_split
        return (
            words,
            torch.LongTensor(token_ids).to(self._arguments_service.device))
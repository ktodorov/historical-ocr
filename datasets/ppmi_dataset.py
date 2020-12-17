from entities.tokens_occurrence_stats import TokensOccurrenceStats
import os
import numpy as np
import torch
import pickle
from overrides import overrides

from datasets.dataset_base import DatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.tokenize.base_tokenize_service import BaseTokenizeService

from services.process.ppmi_process_service import PPMIProcessService

class PPMIDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: PPMIProcessService,
            **kwargs):
        super().__init__()

        self._arguments_service = arguments_service

        self._occurrence_stats: TokensOccurrenceStats = process_service.get_occurrence_stats(ocr_output_type=self._arguments_service.ocr_output_type)

    @overrides
    def __len__(self):
        return 1 # TODO Check

    @overrides
    def __getitem__(self, idx):
        return (self._occurrence_stats.mutual_occurrences, self._occurrence_stats.token_occurrences)

    @overrides
    def use_collate_function(self) -> bool:
        return False
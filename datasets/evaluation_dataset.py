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

    def __len__(self):
        return len(self._target_tokens)

    def __getitem__(self, idx):
        target_token = self._target_tokens[idx]
        return target_token
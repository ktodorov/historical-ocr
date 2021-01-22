from typing import Dict, List
from entities.skip_gram.skip_gram_corpus import SkipGramCorpus
import os
import numpy as np
import torch
import pickle
from overrides import overrides

from datasets.document_dataset_base import DocumentDatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.log_service import LogService

from services.process.skip_gram_process_service import SkipGramProcessService

class SkipGramDataset(DocumentDatasetBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: SkipGramProcessService,
            log_service: LogService,
            **kwargs):
        super().__init__()

        self._arguments_service = arguments_service
        self._log_service = log_service

        self._text_corpus: SkipGramCorpus = process_service.get_text_corpus(ocr_output_type=self._arguments_service.ocr_output_type)
        self._log_service.log_debug(f'Loaded text corpus of size {self._text_corpus.length} in skip gram dataset')

    @overrides
    def __len__(self):
        return self._text_corpus.length

    @overrides
    def __getitem__(self, idx):
        skip_gram_entries = self._text_corpus.get_entries(idx)

        target_tokens = [x.target_token for x in skip_gram_entries]
        context_tokens = [x.context_token for x in skip_gram_entries]

        return (
            torch.LongTensor(context_tokens).to(self._arguments_service.device),
            torch.LongTensor(target_tokens).to(self._arguments_service.device))

    @overrides
    def get_indices_per_document(self) -> Dict[int, List[int]]:
        return self._text_corpus.get_indices_per_document()
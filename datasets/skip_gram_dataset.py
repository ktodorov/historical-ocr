import os
import numpy as np
import torch
import pickle
from overrides import overrides

from entities.cbow_corpus import CBOWCorpus
from entities.skip_gram_entry import SkipGramEntry

from datasets.dataset_base import DatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.file_service import FileService
from services.mask_service import MaskService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.log_service import LogService

from services.process.skip_gram_process_service import SkipGramProcessService

class SkipGramDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: SkipGramProcessService,
            **kwargs):
        super().__init__()

        self._arguments_service = arguments_service

        self._text_corpus = process_service.get_text_corpus(ocr_output_type=self._arguments_service.ocr_output_type)

    @overrides
    def __len__(self):
        return self._text_corpus.length

    @overrides
    def __getitem__(self, idx):
        skip_gram_entry = self._text_corpus.get_entry(idx)
        return (skip_gram_entry.target_token, skip_gram_entry.context_token)

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        return self._pad_and_sort_batch(sequences)

    def _pad_and_sort_batch(self, DataLoaderBatch):
        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        target_tokens, context_tokens = batch_split

        return (
            torch.LongTensor(context_tokens).to(self._arguments_service.device),
            torch.LongTensor(target_tokens).to(self._arguments_service.device))
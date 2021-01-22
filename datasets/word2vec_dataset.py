import os
from typing import Dict, List
import numpy as np
import torch
import pickle
from overrides import overrides

from datasets.document_dataset_base import DocumentDatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.log_service import LogService

from services.process.word2vec_process_service import Word2VecProcessService

class Word2VecDataset(DocumentDatasetBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: Word2VecProcessService,
            log_service: LogService,
            **kwargs):
        super().__init__()

        self._arguments_service = arguments_service
        self._log_service = log_service

        self._text_corpus = process_service.get_text_corpus(ocr_output_type=self._arguments_service.ocr_output_type)
        self._log_service.log_debug(f'Loaded {self._text_corpus.length} entries in word2vec dataset')

    @overrides
    def __len__(self):
        return self._text_corpus.length

    @overrides
    def __getitem__(self, ids):
        skip_gram_entries = self._text_corpus.get_entries(ids)
        batch_size = len(ids)

        target_tokens = [x.target_token for x in skip_gram_entries]
        context_tokens_lists = [x.context_tokens for x in skip_gram_entries]

        lengths = [len(sequence) for sequence in context_tokens_lists]
        max_length = max(lengths)

        padded_contexts = np.zeros((batch_size, max_length), dtype=np.int64)

        for i, l in enumerate(lengths):
            padded_contexts[i][0:l] = context_tokens_lists[i][0:l]

        return (
            torch.from_numpy(padded_contexts).long().to(self._arguments_service.device),
            torch.LongTensor(target_tokens).to(self._arguments_service.device))

    @overrides
    def get_indices_per_document(self) -> Dict[int, List[int]]:
        return self._text_corpus.get_indices_per_document()
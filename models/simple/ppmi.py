from collections import Counter
from services.log_service import LogService
from enums.ocr_output_type import OCROutputType
from services.process.evaluation_process_service import EvaluationProcessService
from services.process.process_service_base import ProcessServiceBase
from entities.word_evaluation import WordEvaluation
from typing import List
from entities.tokens_occurrence_stats import TokensOccurrenceStats
from overrides import overrides
import numpy as np
import math
import torch
from scipy import sparse
from itertools import product
import tqdm

import multiprocessing

from models.model_base import ModelBase

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.vocabulary_service import VocabularyService


# Positive Pointwise Mutual Information
class PPMI(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            log_service: LogService,
            process_service: ProcessServiceBase = None,
            ocr_output_type: OCROutputType = None):
        super().__init__(data_service, arguments_service, log_service)

        self._arguments_service = arguments_service
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service

        if ocr_output_type is not None:
            dataset_string = self._arguments_service.get_dataset_string()
            vocab_key = f'vocab-{dataset_string}-{ocr_output_type.value}'
            self._vocabulary_service.load_cached_vocabulary(vocab_key)

        self._process_service = process_service

        # needed for column intersection during evaluation
        self._common_word_ids: List[int] = self._get_common_word_ids()

        self._initialized = False
        self._ppmi_matrix = sparse.dok_matrix(
            (self._vocabulary_service.vocabulary_size(),
             self._vocabulary_service.vocabulary_size()),
            dtype=np.float32)

    @overrides
    def forward(self, stats: TokensOccurrenceStats):
        if self._initialized:
            return

        result = self._calculate_pmi(
            stats.mutual_occurrences.todense(),
            positive=True)

        self._ppmi_matrix = sparse.dok_matrix(result)

        self._initialized = True

    def _calculate_pmi(
            self,
            matrix,
            positive=True):
        col_totals = matrix.sum(axis=0)
        total = col_totals.sum()
        row_totals = matrix.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        matrix = matrix / expected
        # Silence distracting warnings about log(0):
        with np.errstate(divide='ignore'):
            matrix = np.log(matrix)

        matrix[np.isinf(matrix) | np.isnan(matrix)] = 0.0  # log(0) = 0
        if positive:
            matrix[matrix < 0] = 0.0

        return matrix

    @overrides
    def get_embeddings(self, tokens: List[str], skip_unknown: bool = False) -> List[WordEvaluation]:
        vocab_ids = np.array([np.array([self._vocabulary_service.string_to_id(token)]) for token in tokens])

        embeddings = self._ppmi_matrix[vocab_ids, self._common_word_ids].toarray()

        assert all(not np.isnan(x).any() for x in embeddings)
        assert len(embeddings) == len(vocab_ids)
        assert len(embeddings[0]) == len(self._common_word_ids)

        if skip_unknown:
            unk_vocab_id = self._vocabulary_service.unk_token
            embeddings = [
                x if vocab_ids[i][0] != unk_vocab_id else None for i, x in enumerate(embeddings)]

        return embeddings

    @overrides
    def save(
            self,
            path: str,
            epoch: int,
            iteration: int,
            best_metrics: object,
            resets_left: int,
            name_prefix: str = None,
            save_model_dict: bool = True) -> bool:
        checkpoint_name = self._get_model_name(name_prefix)
        saved = self._data_service.save_python_obj(
            self._ppmi_matrix,
            path,
            checkpoint_name)

        return saved

    def load(
            self,
            path: str,
            name_prefix: str = None,
            name_suffix: str = None,
            load_model_dict: bool = True,
            use_checkpoint_name: bool = True,
            checkpoint_name: str = None):

        if checkpoint_name is None:
            if not use_checkpoint_name:
                checkpoint_name = name_prefix
            else:
                checkpoint_name = self._arguments_service.resume_checkpoint_name
                if checkpoint_name is None:
                    checkpoint_name = self._get_model_name(
                        name_prefix, name_suffix)

        if not self._data_service.python_obj_exists(path, checkpoint_name):
            raise Exception(
                f'PPMI model checkpoint "{checkpoint_name}" not found at "{path}"')

        self._ppmi_matrix = self._data_service.load_python_obj(
            path, checkpoint_name)
        self._initialized = True
        return None

    def _get_common_word_ids(self) -> List[int]:
        if ((not self._arguments_service.evaluate and not self._arguments_service.run_experiments) or
            self._process_service is None or
                not isinstance(self._process_service, EvaluationProcessService)):
            self._log_service.log_debug(f'Skipping loading common token ids')
            return None

        common_words = self._process_service.get_common_words()
        common_word_ids = [self._vocabulary_service.string_to_id(
            common_word) for common_word in common_words]

        self._log_service.log_debug(f'Loaded {len(common_word_ids)} common token ids')
        return common_word_ids

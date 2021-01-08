from collections import Counter
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
            data_service: DataService):
        super().__init__(data_service, arguments_service)

        self._arguments_service = arguments_service
        self._vocabulary_service = vocabulary_service

        self._initialized = False
        self._ppmi_matrix = sparse.dok_matrix(
            (vocabulary_service.vocabulary_size(),
             vocabulary_service.vocabulary_size()),
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

        matrix[np.isinf(matrix)] = 0.0  # log(0) = 0
        if positive:
            matrix[matrix < 0] = 0.0

        return matrix

    @overrides
    def get_embeddings(self, tokens: List[str], vocab_ids: torch.Tensor, skip_unknown: bool = False) -> List[WordEvaluation]:
        pass

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
            checkpoint_name,
            print_success=False)

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

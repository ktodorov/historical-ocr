from entities.tokens_occurrence_stats import TokensOccurrenceStats
from overrides import overrides
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import embedding

from models.model_base import ModelBase

from entities.model_checkpoint import ModelCheckpoint
from entities.metric import Metric
from entities.batch_representation import BatchRepresentation

from sklearn.model_selection import train_test_split

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.process.word2vec_process_service import Word2VecProcessService
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
        self._ppmi_matrix = np.zeros(
            (vocabulary_service.vocabulary_size(), vocabulary_service.vocabulary_size()),
            dtype=np.float32)


    @overrides
    def forward(self, stats):
        if self._initialized:
            return

        (mutual_occurrences, token_occurrences) = stats
        mutual_occurrences = mutual_occurrences.squeeze().numpy()
        token_occurrences = token_occurrences.squeeze().numpy()

        for w_token_idx, _ in self._vocabulary_service.get_vocabulary_tokens():
            for c_token_idx, _ in self._vocabulary_service.get_vocabulary_tokens():
                self._ppmi_matrix[w_token_idx, c_token_idx] = self._calculate_ppmi_score(
                    mutual_occurrences,
                    token_occurrences,
                    w_token_idx,
                    c_token_idx)

        self._initialized = True

    def _calculate_ppmi_score(
        self,
        co_occurrence_matrix,
        occurrences,
        w_token_idx,
        c_token_idx) -> float:

        denominator = occurrences[w_token_idx] * occurrences[c_token_idx]
        if denominator == 0:
            return 0

        division = co_occurrence_matrix[w_token_idx, c_token_idx] / denominator
        if division == 0:
            return 0

        log_score = math.log2(division)

        ppmi_score = max(log_score, 0)
        return ppmi_score

    @overrides
    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        pass

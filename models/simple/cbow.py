from enums.language import Language
import os
from overrides import overrides

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


class CBOW(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            process_service: Word2VecProcessService = None):
        super().__init__(data_service, arguments_service)

        self._arguments_service = arguments_service

        if process_service is not None:
            embedding_size = process_service.get_embedding_size()
            token_matrix = process_service.get_pretrained_matrix()
            embedding_size = token_matrix.shape[-1]
            self._embeddings = nn.Embedding.from_pretrained(
                embeddings=token_matrix,
                freeze=False,
                padding_idx=vocabulary_service.pad_token)
        else:
            embedding_size = self._get_embedding_size(arguments_service.language)
            self._embeddings = nn.Embedding(
                num_embeddings=vocabulary_service.vocabulary_size(),
                embedding_dim=embedding_size,
                padding_idx=vocabulary_service.pad_token)

        self._linear = nn.Linear(
            embedding_size, vocabulary_service.vocabulary_size())

    @overrides
    def forward(self, input_batch, **kwargs):
        context_tokens, targets = input_batch
        embedded_representation = self._embeddings.forward(context_tokens)

        hidden = self._linear.forward(embedded_representation)
        squeezed_hidden = torch.mean(hidden, dim=1)
        output = F.log_softmax(squeezed_hidden, dim=1)

        return (output, targets)

    def _get_embedding_size(self, language: Language):
        if language == Language.English:
            return 300
        elif language == Language.Dutch:
            return 320

        raise NotImplemented()

    @overrides
    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self._embeddings.forward(tokens)
        embeddings_list = embeddings.squeeze().tolist()
        return embeddings_list

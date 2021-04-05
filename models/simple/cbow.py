from services.log_service import LogService
from enums.ocr_output_type import OCROutputType
from entities.word_evaluation import WordEvaluation
from typing import List
from enums.language import Language
import os
from overrides import overrides

import torch
from torch import nn
import torch.nn.functional as F

from models.model_base import ModelBase

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
            log_service: LogService,
            process_service: Word2VecProcessService = None,
            ocr_output_type: OCROutputType = None):
        super().__init__(data_service, arguments_service, log_service)

        self._arguments_service = arguments_service
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service

        if ocr_output_type is not None:
            dataset_string = '-'.join(sorted(self._arguments_service.datasets))
            vocab_key = f'vocab-{dataset_string}-{ocr_output_type.value}'
            self._vocabulary_service.load_cached_vocabulary(vocab_key)

        if process_service is not None:
            self._log_service.log_debug('Process service is provided. Initializing embeddings from a pretrained matrix')
            embedding_size = process_service.get_embedding_size()
            token_matrix = process_service.get_pretrained_matrix()
            embedding_size = token_matrix.shape[-1]
            self._embeddings = nn.Embedding.from_pretrained(
                embeddings=token_matrix,
                freeze=False,
                padding_idx=self._vocabulary_service.pad_token)
        else:
            self._log_service.log_debug('Process service is not provided. Initializing embeddings randomly')
            embedding_size = self._get_embedding_size(arguments_service.language)
            self._embeddings = nn.Embedding(
                num_embeddings=self._vocabulary_service.vocabulary_size(),
                embedding_dim=embedding_size,
                padding_idx=self._vocabulary_service.pad_token)

        self._linear = nn.Linear(
            embedding_size, self._vocabulary_service.vocabulary_size())

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
        elif language == Language.French:
            return 300
        elif language == Language.German:
            return 300

        raise NotImplementedError()

    @overrides
    def get_embeddings(self, tokens: List[str], vocab_ids: torch.Tensor, skip_unknown: bool = False) -> List[WordEvaluation]:
        if vocab_ids is None:
            vocab_ids = torch.Tensor([self._vocabulary_service.string_to_id(token) for token in tokens]).long().to(self._arguments_service.device)

        embeddings = self._embeddings.forward(vocab_ids)
        embeddings_list = embeddings.squeeze().tolist()

        result = [
            WordEvaluation(token, embeddings_list=[
                           embeddings_list[i] if not skip_unknown or self._vocabulary_service.token_exists(token) else None])
            for i, token in enumerate(tokens)]

        return result

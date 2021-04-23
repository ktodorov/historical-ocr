from services.log_service import LogService
from entities.word_evaluation import WordEvaluation
from typing import List
from enums.ocr_output_type import OCROutputType
from enums.language import Language
import os
from overrides import overrides

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import embedding

from models.model_base import ModelBase

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.process.word2vec_process_service import Word2VecProcessService
from services.data_service import DataService
from services.vocabulary_service import VocabularyService


class SkipGram(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            vocabulary_service: VocabularyService,
            data_service: DataService,
            log_service: LogService,
            process_service: Word2VecProcessService = None,
            ocr_output_type: OCROutputType = None,
            pretrained_matrix=None):
        super().__init__(data_service, arguments_service, log_service)

        self._arguments_service = arguments_service
        self._ocr_output_type = ocr_output_type
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service

        if ocr_output_type is not None:
            dataset_string = self._arguments_service.get_dataset_string()
            vocab_key = f'vocab-{dataset_string}-{ocr_output_type.value}'
            self._vocabulary_service.load_cached_vocabulary(vocab_key)

        self._vocabulary_size = self._vocabulary_service.vocabulary_size()

        if pretrained_matrix is not None:
            self._log_service.log_debug(
                'Pretrained matrix provided. Initializing embeddings from it')
            self._embeddings_input = nn.Embedding.from_pretrained(
                embeddings=pretrained_matrix,
                freeze=True,
                padding_idx=self._vocabulary_service.pad_token)

            self._embeddings_context = nn.Embedding.from_pretrained(
                embeddings=pretrained_matrix,
                freeze=True,
                padding_idx=self._vocabulary_service.pad_token)
        elif process_service is not None:
            self._log_service.log_debug(
                'Process service is provided. Initializing embeddings from a pretrained matrix')
            token_matrix = process_service.get_pretrained_matrix()
            embedding_size = token_matrix.shape[-1]
            self._embeddings_input = nn.Embedding.from_pretrained(
                embeddings=token_matrix,
                freeze=False,
                padding_idx=self._vocabulary_service.pad_token)

            self._embeddings_context = nn.Embedding.from_pretrained(
                embeddings=token_matrix,
                freeze=False,
                padding_idx=self._vocabulary_service.pad_token)
        else:
            self._log_service.log_debug(
                'Process service is not provided. Initializing embeddings randomly')
            embedding_size = self._get_embedding_size(
                arguments_service.language)
            self._embeddings_input = nn.Embedding(
                num_embeddings=self._vocabulary_size,
                embedding_dim=embedding_size,
                padding_idx=self._vocabulary_service.pad_token)

            self._embeddings_context = nn.Embedding(
                num_embeddings=self._vocabulary_size,
                embedding_dim=embedding_size,
                padding_idx=self._vocabulary_service.pad_token)

        self._negative_samples = 10
        self._noise_dist = None

    @overrides
    def forward(self, input_batch, **kwargs):
        context_words, input_words = input_batch
        context_size = context_words.size()[1]
        batch_size = input_words.size()[0]

        # computing out loss
        emb_input = self._embeddings_input.forward(input_words).unsqueeze(2)
        emb_context = self._embeddings_context.forward(context_words)

        nwords = torch.FloatTensor(batch_size, context_size * self._negative_samples).uniform_(
            0, self._vocabulary_size - 1).long().to(self._arguments_service.device)
        emb_negative = self._embeddings_context.forward(nwords).neg()
        out_loss = torch.bmm(
            emb_context, emb_input).squeeze().sigmoid().log().mean(1)
        noise_loss = torch.nn.functional.logsigmoid(torch.bmm(emb_negative, emb_input).squeeze(
        )).view(-1, context_size, self._negative_samples).sum(2).mean(1)
        total_loss = -(out_loss + noise_loss).mean()

        return total_loss

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
    def get_embeddings(self, tokens: List[str], skip_unknown: bool = False) -> List[WordEvaluation]:
        vocab_ids = torch.Tensor([self._vocabulary_service.string_to_id(
            token) for token in tokens]).long().to(self._arguments_service.device)

        embeddings = self._embeddings_input.forward(vocab_ids)
        embeddings_list = embeddings.squeeze().tolist()

        if skip_unknown:
            unk_vocab_id = self._vocabulary_service.unk_token
            embeddings_list = [
                x if vocab_ids[i] != unk_vocab_id else None for i, x in enumerate(embeddings_list)]

        return embeddings_list

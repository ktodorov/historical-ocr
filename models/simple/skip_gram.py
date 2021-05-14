from models.embedding.skip_gram_embedding_layer import SkipGramEmbeddingLayer
from services.log_service import LogService
from entities.word_evaluation import WordEvaluation
from typing import List
from enums.ocr_output_type import OCROutputType
from enums.language import Language
import os
from overrides import overrides

import torch
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

        randomly_initialized = False
        freeze_embeddings = True
        if pretrained_matrix is None and process_service is not None:
            pretrained_matrix, randomly_initialized = process_service.get_pretrained_matrix()
            freeze_embeddings = False

        if ocr_output_type is not None:
            dataset_string = self._arguments_service.get_dataset_string()
            vocab_key = f'vocab-{dataset_string}-{ocr_output_type.value}'
            self._vocabulary_service.load_cached_vocabulary(vocab_key)

        self._vocabulary_size = self._vocabulary_service.vocabulary_size()
        self._embedding_layer = SkipGramEmbeddingLayer(
            log_service,
            self._arguments_service.language,
            self._vocabulary_size,
            pretrained_matrix,
            randomly_initialized,
            freeze_embeddings,
            pad_token=self._vocabulary_service.pad_token)

        self._negative_samples = 10

    @overrides
    def forward(self, input_batch, **kwargs):
        context_words, target_words = input_batch
        batch_size = target_words.size()[0]

        emb_context = self._embedding_layer.forward_context(context_words)
        emb_target = self._embedding_layer.forward_target(target_words)

        neg_samples = self._get_negative_samples(batch_size)
        emb_negative = self._embedding_layer.forward_negative(neg_samples)

        return (emb_target, emb_context, emb_negative)

    def _get_negative_samples(self, batch_size: int):
        noise_dist = torch.ones(self._vocabulary_size)
        num_neg_samples_for_this_batch = batch_size * self._negative_samples
        negative_examples = torch.multinomial(
            noise_dist, num_neg_samples_for_this_batch, replacement=True)
        negative_examples = negative_examples.view(
            batch_size, self._negative_samples).to(self._arguments_service.device)
        return negative_examples

    @overrides
    def get_embeddings(self, tokens: List[str], skip_unknown: bool = False) -> List[WordEvaluation]:
        vocab_ids = torch.Tensor([self._vocabulary_service.string_to_id(
            token) for token in tokens]).long().to(self._arguments_service.device)

        embeddings = self._embedding_layer.forward_target(vocab_ids)
        embeddings_list = embeddings.squeeze().tolist()

        if skip_unknown:
            unk_vocab_id = self._vocabulary_service.unk_token
            embeddings_list = [
                x if vocab_ids[i] != unk_vocab_id else None for i, x in enumerate(embeddings_list)]

        return embeddings_list

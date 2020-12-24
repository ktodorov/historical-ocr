import os
from services.file_service import FileService

from overrides import overrides
from torch._C import dtype
import gensim
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

from enums.ocr_output_type import OCROutputType
from enums.language import Language

from entities.cbow_corpus import CBOWCorpus

from services.process.icdar_process_service import ICDARProcessService

from services.download.ocr_download_service import OCRDownloadService
from services.arguments.ocr_quality_non_context_arguments_service import OCRQualityNonContextArgumentsService
from services.cache_service import CacheService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService


class Word2VecProcessService(ICDARProcessService):
    def __init__(
            self,
            ocr_download_service: OCRDownloadService,
            arguments_service: OCRQualityNonContextArgumentsService,
            cache_service: CacheService,
            log_service: LogService,
            vocabulary_service: VocabularyService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService):
        super().__init__(
            ocr_download_service=ocr_download_service,
            arguments_service=arguments_service,
            cache_service=cache_service,
            vocabulary_service=vocabulary_service,
            tokenize_service=tokenize_service)

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._log_service = log_service
        self._vocabulary_service = vocabulary_service
        self._file_service = file_service

    def get_text_corpus(self, ocr_output_type: OCROutputType) -> CBOWCorpus:
        limit_size = self._arguments_service.train_dataset_limit_size
        text_corpus = self._load_text_corpus(ocr_output_type, limit_size)
        return text_corpus

    @overrides
    def _generate_corpora_entries(self, data_ids):
        return CBOWCorpus(data_ids, window_size=2)

    def get_embedding_size(self) -> int:
        if self._arguments_service.language == Language.English:
            return 300
        elif self._arguments_service.language == Language.Dutch:
            return 320

        raise Exception('Unsupported word2vec language')

    def get_pretrained_matrix(self) -> torch.Tensor:
        if not self._vocabulary_service.vocabulary_is_initialized():
            raise Exception('Vocabulary not initialized')

        ocr_output_type = self._arguments_service.ocr_output_type

        random_suffix = ''
        if self._arguments_service.initialize_randomly:
            random_suffix = '-rnd-init'

        token_matrix = self._cache_service.get_item_from_cache(
            item_key=f'word-matrix-{ocr_output_type.value}{random_suffix}',
            callback_function=self._generate_token_matrix)

        token_matrix = token_matrix.to(self._arguments_service.device)
        return token_matrix

    def _generate_token_matrix(self):
        data_path = self._file_service.combine_path(self._file_service.get_challenge_path(
        ), 'word2vec', self._arguments_service.language.value)

        word2vec_model_name, word2vec_binary = self._get_word2vec_model_info()
        word2vec_model_path = os.path.join(data_path, word2vec_model_name)
        word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format(
            word2vec_model_path, binary=word2vec_binary)
        pretrained_weight_matrix = np.random.rand(
            self._vocabulary_service.vocabulary_size(),
            word2vec_weights.vector_size)

        if not self._arguments_service.initialize_randomly:
            vocabulary_items = tqdm(self._vocabulary_service.get_vocabulary_tokens(
            ), desc="Generating pre-trained matrix", total=self._vocabulary_service.vocabulary_size())
            for (index, token) in vocabulary_items:
                if token in word2vec_weights.vocab:
                    pretrained_weight_matrix[index] = word2vec_weights.wv[token]

        result = torch.from_numpy(pretrained_weight_matrix).float()
        return result

    def _get_word2vec_model_info(self) -> Tuple[str, bool]:
        if self._arguments_service.language == Language.English:
            return 'GoogleNews-vectors-negative300.bin', True
        elif self._arguments_service.language == Language.Dutch:
            return 'combined-320.txt', False

        raise Exception('Unsupported word2vec language')

    def _load_text_corpus(
            self,
            ocr_output_type: OCROutputType,
            reduction: int) -> CBOWCorpus:
        corpus = self._cache_service.get_item_from_cache(
            item_key=f'word2vec-data-{ocr_output_type.value}-ws-2',
            callback_function=self._generate_ocr_corpora)

        total_amount = corpus.length
        if reduction is not None:
            corpus.cut_data(reduction)

        print(
            f'Loaded {corpus.length:,} entries out of {total_amount:,} total for {ocr_output_type.value}')
        self._log_service.log_summary(
            key=f'\'{ocr_output_type.value}\' entries amount', value=corpus.length)

        return corpus
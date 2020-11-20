import os
from services.file_service import FileService

from torch._C import dtype
import gensim
import numpy as np
import torch
from tqdm import tqdm

from enums.ocr_output_type import OCROutputType

from entities.cbow_corpus import CBOWCorpus

from services.process.process_service_base import ProcessServiceBase

from services.download.ocr_download_service import OCRDownloadService
from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.cache_service import CacheService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService

class Word2VecProcessService(ProcessServiceBase):
    def __init__(
            self,
            ocr_download_service: OCRDownloadService,
            arguments_service: ArgumentsServiceBase,
            cache_service: CacheService,
            log_service: LogService,
            vocabulary_service: VocabularyService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService):

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._ocr_download_service = ocr_download_service
        self._log_service = log_service
        self._vocabulary_service = vocabulary_service
        self._file_service = file_service

        self._tokenize_service = tokenize_service


    def get_text_corpus(self, ocr_output_type: OCROutputType) -> CBOWCorpus:
        limit_size = self._arguments_service.train_dataset_limit_size

        text_corpus = self._load_text_corpus(
            ocr_output_type,
            limit_size)

        return text_corpus

    def _generate_ocr_corpora(self):
        self._ocr_download_service.download_data(self._arguments_service.language)

        ocr_data, gs_data = self._cache_service.get_item_from_cache(
            item_key='train-validation-pairs',
            callback_function=self._read_data)

        tokenized_ocr_data = self._tokenize_service.tokenize_sequences(ocr_data)
        tokenized_gs_data = self._tokenize_service.tokenize_sequences(gs_data)

        self._vocabulary_service.initialize_vocabulary_from_corpus(tokenized_ocr_data + tokenized_gs_data)

        ocr_data_ids = [self._vocabulary_service.string_to_ids(x) for x in tokenized_ocr_data]
        gs_data_ids = [self._vocabulary_service.string_to_ids(x) for x in tokenized_gs_data]

        ocr_corpus = CBOWCorpus(ocr_data_ids, window_size=2)
        gs_corpus = CBOWCorpus(gs_data_ids, window_size=2)

        return (ocr_corpus, gs_corpus)

    def get_pretrained_matrix(self) -> torch.Tensor:
        token_matrix = self._cache_service.get_item_from_cache(
            item_key='word-matrix',
            callback_function=self._generate_token_matrix)

        token_matrix = token_matrix.to(self._arguments_service.device)
        return token_matrix

    def _generate_token_matrix(self):
        data_path = self._file_service.get_data_path()
        word2vec_model_path = os.path.join(data_path, 'GoogleNews-vectors-negative300.bin')
        word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary = True)
        pretrained_weight_matrix = np.random.rand(
            self._vocabulary_service.vocabulary_size(),
            word2vec_weights.vector_size)

        vocabulary_items = tqdm(self._vocabulary_service.get_vocabulary_tokens(), desc="Generating pre-trained matrix", total=self._vocabulary_service.vocabulary_size())
        for (index, token) in vocabulary_items:
            if token in word2vec_weights.vocab:
                pretrained_weight_matrix[index] = word2vec_weights.wv[token]

        result = torch.from_numpy(pretrained_weight_matrix).float()
        return result

    def _load_text_corpus(
            self,
            ocr_output_type: OCROutputType,
            reduction: int) -> CBOWCorpus:
        (ocr_corpus, gs_corpus) = self._cache_service.get_item_from_cache(
            item_key=f'word2vec-data-ws-2',
            callback_function=self._generate_ocr_corpora)

        corpus = ocr_corpus if ocr_output_type == OCROutputType.Raw else gs_corpus

        total_amount = corpus.length
        if reduction is not None:
            corpus.cut_data(reduction)

        print(
            f'Loaded {corpus.length:,} entries out of {total_amount:,} total for {ocr_output_type.value}')
        self._log_service.log_summary(
            key=f'\'{ocr_output_type.value}\' entries amount', value=corpus.length)

        return corpus

    def _load_file_data(self):
        cache_keys = [
            'trove-dataset',
            'newseye-2017-full-dataset',
            'newseye-2019-train-dataset',
            'newseye-2019-eval-dataset']

        number_of_files = len(cache_keys)

        ocr_file_data = []
        gs_file_data = []

        for i, cache_key in enumerate(cache_keys):
            print(f'{i}/{number_of_files}             \r', end='')
            result = self._cache_service.get_item_from_cache(cache_key)
            if result is None:
                continue

            ocr_file_data.extend(result[0])
            gs_file_data.extend(result[1])

        return ocr_file_data, gs_file_data

    def _read_data(self):
        ocr_gs_file_data_cache_key = f'ocr-gs-file-data'
        ocr_file_data, gs_file_data = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_file_data_cache_key,
            callback_function=self._load_file_data)

        return ocr_file_data, gs_file_data
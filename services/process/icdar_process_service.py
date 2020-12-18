import os
from services.file_service import FileService

from torch._C import dtype
import gensim
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

from enums.ocr_output_type import OCROutputType
from enums.language import Language

from entities.cbow_corpus import CBOWCorpus

from services.process.process_service_base import ProcessServiceBase

from services.download.ocr_download_service import OCRDownloadService
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.cache_service import CacheService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService


class ICDARProcessService(ProcessServiceBase):
    def __init__(
            self,
            ocr_download_service: OCRDownloadService,
            arguments_service: OCRQualityArgumentsService,
            cache_service: CacheService,
            vocabulary_service: VocabularyService,
            tokenize_service: BaseTokenizeService,
            min_occurrence_limit: int = None):

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._ocr_download_service = ocr_download_service
        self._vocabulary_service = vocabulary_service
        self._tokenize_service = tokenize_service

        self._min_occurrence_limit = min_occurrence_limit
        self._vocab_key = f'vocab-{arguments_service.ocr_output_type.value}'

        if not self._vocabulary_service.load_cached_vocabulary(self._vocab_key):
            self._initialize_vocabulary()

    def _initialize_vocabulary(self):
        self._ocr_download_service.download_data(
            self._arguments_service.language)

        ocr_data, gs_data = self._cache_service.get_item_from_cache(
            item_key='train-validation-data',
            callback_function=self._read_data)

        tokenized_data = self._tokenize_service.tokenize_sequences(
            gs_data if self._arguments_service.ocr_output_type == OCROutputType.GroundTruth else ocr_data
        )

        self._vocabulary_service.initialize_vocabulary_from_corpus(
            tokenized_data, 
            min_occurrence_limit=self._min_occurrence_limit,
            vocab_key=self._vocab_key)

    def _generate_ocr_corpora(self):
        (ocr_data, gs_data) = self._cache_service.get_item_from_cache(
            item_key='train-validation-data')

        tokenized_ocr_data = self._tokenize_service.tokenize_sequences(
            ocr_data)
        tokenized_gs_data = self._tokenize_service.tokenize_sequences(gs_data)

        self._save_common_words(tokenized_ocr_data, tokenized_gs_data)

        ocr_output_type = self._arguments_service.ocr_output_type
        data_ids = [self._vocabulary_service.string_to_ids(
            x) for x in (tokenized_ocr_data if ocr_output_type == OCROutputType.Raw else tokenized_gs_data)]

        self._cache_service.cache_item(
            item_key=f'token-ids-{self._arguments_service.ocr_output_type.value}',
            item=data_ids)

        result = self._generate_corpora_entries(data_ids)
        return result

    def _generate_corpora_entries(self, data_ids):
        return None

    def _save_common_words(self, tokenized_ocr_data: List[List[str]], tokenized_gs_data: List[List[str]]):
        ocr_unique_tokens = set(
            [item for sublist in tokenized_ocr_data for item in sublist])
        gs_unique_tokens = set(
            [item for sublist in tokenized_gs_data for item in sublist])

        common_tokens = list(ocr_unique_tokens & gs_unique_tokens)
        self._cache_service.cache_item(
            item_key=f'common-tokens-{self._arguments_service.language.value}',
            item=common_tokens,
            configuration_specific=False)

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
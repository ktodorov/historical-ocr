from entities.cache.cache_options import CacheOptions
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

from services.process.process_service_base import ProcessServiceBase

from services.download.ocr_download_service import OCRDownloadService
from services.arguments.ocr_quality_non_context_arguments_service import OCRQualityNonContextArgumentsService
from services.cache_service import CacheService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService


class ICDARProcessService(ProcessServiceBase):
    def __init__(
            self,
            ocr_download_service: OCRDownloadService,
            arguments_service: OCRQualityNonContextArgumentsService,
            cache_service: CacheService,
            vocabulary_service: VocabularyService,
            tokenize_service: BaseTokenizeService,
            log_service: LogService):

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._ocr_download_service = ocr_download_service
        self._vocabulary_service = vocabulary_service
        self._tokenize_service = tokenize_service
        self._log_service = log_service

        self._min_occurrence_limit = self._arguments_service.minimal_occurrence_limit
        self._vocab_key = f'vocab-{self._get_dataset_string()}-{arguments_service.ocr_output_type.value}'

        if not self._vocabulary_service.load_cached_vocabulary(self._vocab_key):
            self._log_service.log_debug(
                'Vocabulary was not loaded. Attempting to initialize...')
            self._initialize_vocabulary()
        else:
            self._log_service.log_debug('Vocabulary loaded successfully')

    def _initialize_vocabulary(self):
        self._ocr_download_service.download_data(
            self._arguments_service.language)

        ocr_data, gs_data = self._read_data()
        tokenized_data = self._tokenize_service.tokenize_sequences(
            gs_data if self._arguments_service.ocr_output_type == OCROutputType.GroundTruth else ocr_data
        )
        self._log_service.log_debug(
            f'Tokenized {len(tokenized_data)} strings successfully')

        self._vocabulary_service.initialize_vocabulary_from_corpus(
            tokenized_data,
            min_occurrence_limit=self._min_occurrence_limit,
            vocab_key=self._vocab_key)

    def _generate_ocr_corpora(self):
        ocr_data, gs_data = self._read_data()
        tokenized_ocr_data = self._tokenize_service.tokenize_sequences(
            ocr_data)
        tokenized_gs_data = self._tokenize_service.tokenize_sequences(gs_data)

        self._save_common_tokens(tokenized_ocr_data, tokenized_gs_data)

        ocr_output_type = self._arguments_service.ocr_output_type
        data_ids = [self._vocabulary_service.string_to_ids(
            x) for x in (tokenized_ocr_data if ocr_output_type == OCROutputType.Raw else tokenized_gs_data)]

        result = self._generate_corpora_entries(data_ids)
        return result

    def _generate_corpora_entries(self, data_ids):
        return None

    def _save_common_tokens(self, tokenized_ocr_data: List[List[str]], tokenized_gs_data: List[List[str]]):
        """Saves the intersection of the tokens from both output types, as well as the ids of these tokens for the current output type

        :param tokenized_ocr_data: The tokenized data for OCR output type
        :type tokenized_ocr_data: List[List[str]]
        :param tokenized_gs_data: The tokenized data for GT output type
        :type tokenized_gs_data: List[List[str]]
        """
        self._log_service.log_debug('Saving common tokens')
        token_pairs_cache_key = f'common-t-pairs-{self._get_dataset_string()}-{self._arguments_service.ocr_output_type.value}-lim-{self._arguments_service.minimal_occurrence_limit}'
        if self._cache_service.item_exists(CacheOptions(token_pairs_cache_key)):
            return

        common_tokens = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'common-tokens-{self._get_dataset_string()}',
                configuration_specific=False),
            callback_function=lambda: self._combine_common_words(tokenized_ocr_data, tokenized_gs_data))

        token_id_pairs = []
        for common_token in common_tokens:
            token_ids = [self._vocabulary_service.string_to_id(common_token)]
            if token_ids[0] == self._vocabulary_service.unk_token:
                token_ids = None

            token_id_pairs.append((common_token, token_ids))

        self._cache_service.cache_item(
            token_id_pairs,
            CacheOptions(token_pairs_cache_key))

        self._log_service.log_debug(
            f'Saved {len(token_id_pairs)} common token pairs successfully')

    def _combine_common_words(self, tokenized_ocr_data: List[List[str]], tokenized_gs_data: List[List[str]]):
        ocr_unique_tokens = set(
            [item for sublist in tokenized_ocr_data for item in sublist])
        gs_unique_tokens = set(
            [item for sublist in tokenized_gs_data for item in sublist])

        common_tokens = list(ocr_unique_tokens & gs_unique_tokens)
        return common_tokens

    def _load_file_data(self):
        number_of_files = len(self._arguments_service.datasets)

        ocr_file_data = []
        gs_file_data = []

        for i, dataset in enumerate(self._arguments_service.datasets):
            print(f'{i}/{number_of_files}             \r', end='')
            result = self._ocr_download_service.get_downloaded_dataset(dataset)
            if result is None:
                self._log_service.log_debug(
                    f'Did not find \'{dataset}\' dataset to load')
                continue
            else:
                self._log_service.log_debug(f'Loading \'{dataset}\' data')

            ocr_file_data.extend(result[0])
            gs_file_data.extend(result[1])

        return ocr_file_data, gs_file_data

    def _read_data(self):
        ocr_file_data, gs_file_data = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'ocr-gs-file-data-{self._get_dataset_string()}',
                configuration_specific=False),
            callback_function=self._load_file_data)

        return ocr_file_data, gs_file_data

    def _get_dataset_string(self):
        return '-'.join(sorted(self._arguments_service.datasets))
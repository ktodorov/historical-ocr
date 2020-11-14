from typing import List
import random

from enums.ocr_output_type import OCROutputType

from entities.language_data import LanguageData

from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.process.process_service_base import ProcessServiceBase
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.download.ocr_download_service import OCRDownloadService

from services.vocabulary_service import VocabularyService
from services.cache_service import CacheService
from services.log_service import LogService


class TransformerProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            ocr_download_service: OCRDownloadService,
            tokenize_service: BaseTokenizeService,
            cache_service: CacheService,
            log_service: LogService):
        super().__init__()

        self._arguments_service = arguments_service
        self._tokenize_service = tokenize_service
        self._ocr_download_service = ocr_download_service
        self._cache_service = cache_service
        self._log_service = log_service

    def get_language_data(self):
        language_data: LanguageData = None
        limit_size = self._arguments_service.train_dataset_limit_size

        language_data = self._load_language_data(
            limit_size)

        return language_data

    def _generate_language_data(self):
        self._ocr_download_service.download_data(self._arguments_service.language, max_string_length=500)

        pairs = self._cache_service.get_item_from_cache(
            item_key='train-validation-pairs',
            callback_function=self._read_data)

        language_data = LanguageData.from_pairs(
            self._tokenize_service,
            pairs)

        return language_data

    def _load_language_data(
            self,
            reduction: int) -> LanguageData:
        language_data = self._cache_service.get_item_from_cache(
            item_key=f'language-data',
            callback_function=self._generate_language_data)

        total_amount = language_data.length
        if reduction is not None:
            language_data.cut_data(reduction)

        print(
            f'Loaded {language_data.length} entries out of {total_amount} total')
        self._log_service.log_summary(
            key=f'entries amount', value=language_data.length)

        return language_data

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
        (ocr_file_data, gs_file_data) = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_file_data_cache_key,
            callback_function=self._load_file_data)

        decoded_pairs_cache_key = f'metrics-decoded-pairs'
        decoded_pairs = self._cache_service.get_item_from_cache(
            item_key=decoded_pairs_cache_key,
            callback_function=lambda: (list(zip(ocr_file_data, gs_file_data))))

        return decoded_pairs
from typing import List
import random

from enums.ocr_output_type import OCROutputType

from entities.transformer_entry import TransformerEntry

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

    def get_entries(self, ocr_output_type: OCROutputType):
        entries = None
        limit_size = self._arguments_service.train_dataset_limit_size

        entries = self._load_transformer_entries(
            ocr_output_type,
            limit_size)

        return entries

    def _generate_entries(self):
        self._ocr_download_service.download_data(self._arguments_service.language, max_string_length=500)

        ocr_file_data, gs_file_data = self._cache_service.get_item_from_cache(
            item_key='train-validation-data',
            callback_function=self._read_data)

        encoded_ocr_sequences = self._tokenize_service.encode_sequences(ocr_file_data)
        encoded_gs_sequences = self._tokenize_service.encode_sequences(gs_file_data)

        ocr_entries = [TransformerEntry(ids, special_tokens_mask) for ids, _, _, special_tokens_mask in encoded_ocr_sequences]
        gs_entries = [TransformerEntry(ids, special_tokens_mask) for ids, _, _, special_tokens_mask in encoded_gs_sequences]

        ocr_ids = [x.token_ids for x in ocr_entries]
        gs_ids = [x.token_ids for x in gs_entries]

        self._cache_service.cache_item(
            item_key='token-ids',
            item=(ocr_ids, gs_ids))

        return ocr_entries, gs_entries

    def _load_transformer_entries(
            self,
            ocr_output_type: OCROutputType,
            reduction: int) -> List[TransformerEntry]:
        ocr_entries, gs_entries = self._cache_service.get_item_from_cache(
            item_key=f'entries',
            callback_function=self._generate_entries)

        entries = ocr_entries if ocr_output_type == OCROutputType.Raw else gs_entries

        total_amount = len(entries)
        if reduction is not None:
            entries = entries[:reduction]

        print(
            f'Loaded {len(entries)} entries out of {total_amount} total')
        self._log_service.log_summary(
            key=f'entries amount', value=len(entries))

        return entries

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

        return ocr_file_data, gs_file_data
from typing import List, Tuple
import random

from enums.ocr_output_type import OCROutputType

from entities.transformers.transformer_entry import TransformerEntry

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
        self._ocr_download_service.download_data(
            self._arguments_service.language, max_string_length=500)

        ocr_file_data, gs_file_data = self._cache_service.get_item_from_cache(
            item_key='train-validation-data',
            callback_function=self._read_data)

        encoded_ocr_sequences = self._tokenize_service.encode_sequences(
            ocr_file_data)
        encoded_gs_sequences = self._tokenize_service.encode_sequences(
            gs_file_data)

        self._save_common_tokens()

        ocr_entries = [TransformerEntry(i, ids, special_tokens_mask)
                       for i, (ids, _, _, special_tokens_mask) in enumerate(encoded_ocr_sequences)]
        gs_entries = [TransformerEntry(i, ids, special_tokens_mask)
                      for i, (ids, _, _, special_tokens_mask) in enumerate(encoded_gs_sequences)]

        ocr_ids = [x.token_ids for x in ocr_entries]
        gs_ids = [x.token_ids for x in gs_entries]

        self._cache_service.cache_item(
            item_key='token-ids',
            item=(ocr_ids, gs_ids))

        return ocr_entries, gs_entries

    def _save_common_tokens(self):
        self._log_service.log_debug('Saving common tokens')
        token_pairs_cache_key = f'common-token-pairs-{self._arguments_service.ocr_output_type.value}'
        if self._cache_service.item_exists(token_pairs_cache_key):
            return

        common_tokens_cache_key = f'common-tokens-{self._arguments_service.language.value}'
        common_tokens = self._cache_service.get_item_from_cache(
            item_key=common_tokens_cache_key,
            configuration_specific=False)

        token_id_pairs = []
        for common_token in common_tokens:
            token_ids, _, _, _ = self._tokenize_service.encode_sequence(common_token)
            token_id_pairs.append((common_token, token_ids))

        self._cache_service.cache_item(
            item_key=token_pairs_cache_key,
            item=token_id_pairs)

        self._log_service.log_debug(
            f'Saved {len(token_id_pairs)} common token pairs successfully')

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

        self._log_service.log_info(
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
                self._log_service.log_debug(
                    f'Did not find \'{cache_key}\' data to load')
                continue
            else:
                self._log_service.log_debug(f'Loading \'{cache_key}\' data')

            ocr_file_data.extend(result[0])
            gs_file_data.extend(result[1])

        return ocr_file_data, gs_file_data

    def _read_data(self):
        ocr_gs_file_data_cache_key = f'ocr-gs-file-data'
        (ocr_file_data, gs_file_data) = self._cache_service.get_item_from_cache(
            item_key=ocr_gs_file_data_cache_key,
            callback_function=self._load_file_data)

        return ocr_file_data, gs_file_data

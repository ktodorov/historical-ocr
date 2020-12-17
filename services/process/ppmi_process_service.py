import os
from services.file_service import FileService

from torch._C import dtype
import gensim
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple
from entities.tokens_occurrence_stats import TokensOccurrenceStats

from overrides import overrides

from enums.ocr_output_type import OCROutputType
from services.download.ocr_download_service import OCRDownloadService
from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.cache_service import CacheService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.process.icdar_process_service import ICDARProcessService

class PPMIProcessService(ICDARProcessService):
    def __init__(
            self,
            ocr_download_service: OCRDownloadService,
            arguments_service: ArgumentsServiceBase,
            cache_service: CacheService,
            vocabulary_service: VocabularyService,
            tokenize_service: BaseTokenizeService):
        super().__init__(
            ocr_download_service=ocr_download_service,
            arguments_service=arguments_service,
            cache_service=cache_service,
            vocabulary_service=vocabulary_service,
            tokenize_service=tokenize_service,
            min_occurrence_limit=5)

    def get_occurrence_stats(self, ocr_output_type: OCROutputType) -> TokensOccurrenceStats:
        occurrence_stats = self._load_occurrence_stats(ocr_output_type)
        return occurrence_stats

    def _load_occurrence_stats(self, ocr_output_type: OCROutputType) -> TokensOccurrenceStats:
        (raw_stats, gs_stats) = self._cache_service.get_item_from_cache(
            item_key=f'tokens-occurrences-stats',
            callback_function=self._generate_ocr_corpora)

        stats: TokensOccurrenceStats = raw_stats if ocr_output_type == OCROutputType.Raw else gs_stats

        # total_amount = corpus.length

        # print(
        #     f'Loaded {corpus.length:,} entries out of {total_amount:,} total for {ocr_output_type.value}')
        # self._log_service.log_summary(
        #     key=f'\'{ocr_output_type.value}\' entries amount', value=corpus.length)

        return stats

    @overrides
    def _generate_corpora_entries(self, ocr_data_ids, gs_data_ids):
        raw_stats = TokensOccurrenceStats(ocr_data_ids, self._vocabulary_service.vocabulary_size())
        gs_stats = TokensOccurrenceStats(gs_data_ids, self._vocabulary_service.vocabulary_size())
        return (raw_stats, gs_stats)
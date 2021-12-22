from entities.cache.cache_options import CacheOptions
import os
from services.file_service import FileService

from torch._C import dtype
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
            tokenize_service: BaseTokenizeService,
            log_service: LogService):
        super().__init__(
            ocr_download_service=ocr_download_service,
            arguments_service=arguments_service,
            cache_service=cache_service,
            vocabulary_service=vocabulary_service,
            tokenize_service=tokenize_service,
            log_service=log_service)

    def get_occurrence_stats(self, ocr_output_type: OCROutputType) -> TokensOccurrenceStats:
        occurrence_stats: TokensOccurrenceStats = self._cache_service.get_item_from_cache(
            CacheOptions(f'tokens-occurrences-stats-{ocr_output_type.value}'),
            callback_function=self._generate_ocr_corpora)

        return occurrence_stats

    def _generate_corpora_entries(self, data_ids):
        token_stats = TokensOccurrenceStats(data_ids, self._vocabulary_service.vocabulary_size())
        return token_stats
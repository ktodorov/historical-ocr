import os
from services.file_service import FileService

from torch._C import dtype
import gensim
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

from overrides import overrides

from enums.ocr_output_type import OCROutputType
from enums.language import Language

from entities.skip_gram_corpus import SkipGramCorpus

from services.process.process_service_base import ProcessServiceBase

from services.download.ocr_download_service import OCRDownloadService
from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.cache_service import CacheService
from services.log_service import LogService
from services.vocabulary_service import VocabularyService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.process.word2vec_process_service import Word2VecProcessService

class SkipGramProcessService(Word2VecProcessService):
    def __init__(
            self,
            ocr_download_service: OCRDownloadService,
            arguments_service: ArgumentsServiceBase,
            cache_service: CacheService,
            log_service: LogService,
            vocabulary_service: VocabularyService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService):

        super().__init__(
            ocr_download_service,
            arguments_service,
            cache_service,
            log_service,
            vocabulary_service,
            file_service,
            tokenize_service)

    @overrides
    def get_text_corpus(self, ocr_output_type: OCROutputType) -> SkipGramCorpus:
        limit_size = self._arguments_service.train_dataset_limit_size

        text_corpus = self._load_text_corpus(
            ocr_output_type,
            limit_size)

        return text_corpus

    @overrides
    def _load_text_corpus(
            self,
            ocr_output_type: OCROutputType,
            reduction: int) -> SkipGramCorpus:
        (ocr_corpus, gs_corpus) = self._cache_service.get_item_from_cache(
            item_key=f'skip-gram-data',
            callback_function=self._generate_ocr_corpora)

        corpus: SkipGramCorpus = ocr_corpus if ocr_output_type == OCROutputType.Raw else gs_corpus

        total_amount = corpus.length
        if reduction is not None:
            corpus.cut_data(reduction)

        print(
            f'Loaded {corpus.length:,} entries out of {total_amount:,} total for {ocr_output_type.value}')
        self._log_service.log_summary(
            key=f'\'{ocr_output_type.value}\' entries amount', value=corpus.length)

        return corpus

    @overrides
    def _generate_corpora_entries(self, ocr_data_ids, gs_data_ids):
        ocr_corpus = SkipGramCorpus(ocr_data_ids)
        gs_corpus = SkipGramCorpus(gs_data_ids)
        return (ocr_corpus, gs_corpus)
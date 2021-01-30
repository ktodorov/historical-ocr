from entities.cache.cache_options import CacheOptions
from enums.part_of_speech import PartOfSpeech
from services.tagging_service import TaggingService
from services.arguments.ocr_quality_non_context_arguments_service import OCRQualityNonContextArgumentsService
from enums.ocr_output_type import OCROutputType
from typing import Any, Dict, List, Tuple

from entities.token_representation import TokenRepresentation

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.process.process_service_base import ProcessServiceBase
from services.tokenize.base_tokenize_service import BaseTokenizeService

from services.vocabulary_service import VocabularyService
from services.cache_service import CacheService
from services.log_service import LogService


class EvaluationProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            cache_service: CacheService,
            log_service: LogService,
            vocabulary_service: VocabularyService,
            tokenize_service: BaseTokenizeService,
            tagging_service: TaggingService):
        super().__init__()

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._log_service = log_service
        self._vocabulary_service = vocabulary_service
        self._tokenize_service = tokenize_service
        self._tagging_service = tagging_service

    def get_target_tokens(self, pos_tags: List[PartOfSpeech] = None) -> List[TokenRepresentation]:
        common_tokens_information: Dict[str, List[List[int]]] = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'common-tokens-information-lim{self._arguments_service.minimal_occurrence_limit}'),
            callback_function=self.get_common_words)

        self._log_service.log_info(
            f'Loaded {len(common_tokens_information)} common words')

        if pos_tags is not None:
            common_tokens_information = {
                token: value
                for token, value in common_tokens_information.items()
                if self._tagging_service.get_part_of_speech_tag(token) in pos_tags
            }

            self._log_service.log_info(
                f'Filtered common words. Left with {len(common_tokens_information)} common words')

        result = [
            TokenRepresentation(
                token=token,
                vocabulary_ids=token_vocab_ids)
            for token, token_vocab_ids
            in common_tokens_information.items()
        ]

        return result

    def get_common_words(self) -> Dict[str, List[List[int]]]:
        common_tokens = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'common-tokens-{self._arguments_service.language.value}',
                configuration_specific=False))

        result: Dict[str, List[List[int]]] = {x: [] for x in common_tokens}

        for ocr_output_type in [OCROutputType.Raw, OCROutputType.GroundTruth]:
            limit_suffix = ''
            if self._arguments_service.minimal_occurrence_limit is not None:
                limit_suffix = f'-lim-{self._arguments_service.minimal_occurrence_limit}'

            common_token_pairs: List[Tuple[str, List[int]]] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'common-token-pairs-{ocr_output_type.value}{limit_suffix}'))

            if common_token_pairs is None:
                error_message = f'Token pairs not found for OCR output type \'{ocr_output_type.value}\''
                self._log_service.log_error(error_message)
                raise Exception(error_message)

            for word, vocab_ids in common_token_pairs:
                if word not in result.keys():
                    continue  # means the word was UNK in another vocabulary

                if vocab_ids is None:
                    # means the word is UNK in this vocabulary
                    del result[word]
                    continue

                result[word].append(vocab_ids)

        result = self._intersect_words(result)
        return result

    def _intersect_words(
            self,
            current_result: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        common_tokens_all_configs = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'common-tokens-{self._arguments_service.language.value}-all-config',
                configuration_specific=False),
            callback_function=lambda: {})

        common_tokens_all_configs[self._arguments_service.configuration] = list(
            current_result.keys())

        self._cache_service.cache_item(
            common_tokens_all_configs,
            CacheOptions(
                f'common-tokens-{self._arguments_service.language.value}-all-config',
                configuration_specific=False))

        all_words_per_config = list(common_tokens_all_configs.values())
        words_intersection = set(all_words_per_config[0]).intersection(
            *all_words_per_config)

        result = {k: v for k, v in current_result.items()
                  if k in words_intersection}
        return result

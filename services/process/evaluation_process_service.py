from enums.configuration import Configuration
from entities.cache.cache_options import CacheOptions
from enums.part_of_speech import PartOfSpeech
from services.tagging_service import TaggingService
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
        common_tokens = self.get_common_words()

        self._log_service.log_info(
            f'Loaded {len(common_tokens)} common words')

        if pos_tags is not None:
            common_tokens = [
                token
                for token in common_tokens
                if self._tagging_service.get_part_of_speech_tag(token) in pos_tags
            ]

            self._log_service.log_info(
                f'Filtered common words. Left with {len(common_tokens)} common words')

        return common_tokens

    def get_common_words(self) -> Dict[str, List[List[int]]]:
        configurations_to_skip = [
            Configuration.BERT,
            Configuration.XLNet,
            Configuration.BART,
            Configuration.RoBERTa,
            Configuration.GloVe,
            Configuration.ALBERT]

        result = None

        for config in Configuration:
            # some configurations do not work with vocabularies or support all words
            if config in configurations_to_skip:
                continue

            # get the config vocabularies
            config_raw_vocabulary = self._cache_service.get_item_from_cache(
                CacheOptions(
                    'vocab',
                    key_suffixes=[
                        '-',
                        self._arguments_service.get_dataset_string(),
                        '-',
                        OCROutputType.Raw.value
                    ],
                    configuration=config))

            config_gt_vocabulary = self._cache_service.get_item_from_cache(
                CacheOptions(
                    'vocab',
                    key_suffixes=[
                        '-',
                        self._arguments_service.get_dataset_string(),
                        '-',
                        OCROutputType.GroundTruth.value
                    ],
                    configuration=config))

            if config_raw_vocabulary is None or config_gt_vocabulary is None:
                self._log_service.log_warning(
                    f'Configuration {config.value} does not have both vocabularies initialized')
                continue

            # extract the tokens from the vocabularies
            raw_tokens = list(config_raw_vocabulary[0].keys())[4:]
            gt_tokens = list(config_gt_vocabulary[0].keys())[4:]

            # intersect
            intersected_tokens = list(set(raw_tokens) & set(gt_tokens))

            self._log_service.log_debug(
                f'Configuration {config.value} tokens intersection - [raw: {len(raw_tokens)}; GT: {len(gt_tokens)}; intersected: {len(intersected_tokens)}')

            # update the current result
            if result is None:
                result = intersected_tokens
            else:
                result = list(set(result) & set(intersected_tokens))

        return result

    def _intersect_words(
            self,
            current_result: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        common_tokens_config_cache_options = CacheOptions(
            f'common-tokens-{self._get_dataset_string()}-all-config',
            configuration_specific=False)

        common_tokens_all_configs = self._cache_service.get_item_from_cache(
            common_tokens_config_cache_options,
            callback_function=lambda: {})

        common_tokens_all_configs[self._arguments_service.configuration] = list(
            current_result.keys())

        self._cache_service.cache_item(
            common_tokens_all_configs,
            common_tokens_config_cache_options)

        all_words_per_config = list(common_tokens_all_configs.values())
        words_intersection = set(all_words_per_config[0]).intersection(
            *all_words_per_config)

        result = {k: v for k, v in current_result.items()
                  if k in words_intersection}
        return result

    def _get_dataset_string(self):
        return '-'.join(sorted(self._arguments_service.datasets))

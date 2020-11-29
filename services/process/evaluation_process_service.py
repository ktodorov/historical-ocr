from typing import List, Tuple

from entities.token_representation import TokenRepresentation

from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.process.process_service_base import ProcessServiceBase
from services.tokenize.base_tokenize_service import BaseTokenizeService

from services.vocabulary_service import VocabularyService
from services.cache_service import CacheService
from services.log_service import LogService

class EvaluationProcessService(ProcessServiceBase):
    def __init__(
        self,
        arguments_service: OCRQualityArgumentsService,
        cache_service: CacheService,
        log_service: LogService,
        vocabulary_service: VocabularyService,
        tokenize_service: BaseTokenizeService):
        super().__init__()

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._log_service = log_service
        self._vocabulary_service = vocabulary_service
        self._tokenize_service = tokenize_service


    def get_target_tokens(self) -> List[TokenRepresentation]:
        (common_words, common_word_ids) = self._cache_service.get_item_from_cache(
            item_key='common-words',
            callback_function=self.get_common_words)

        print(f'Found {len(common_word_ids)} common words')

        result = []
        for word_id, word in zip(common_word_ids, common_words):
            result.append(
                TokenRepresentation(
                    word=word,
                    vocabulary_id=word_id))

        return result

    def get_common_words(self) -> List[int]:
        common_words = self._cache_service.get_item_from_cache(
            item_key=f'common-tokens-{self._arguments_service.language.value}',
            configuration_specific=False)

        if common_words is None:
            return []

        encoded_sequences = self._tokenize_service.encode_sequences(common_words)
        common_ids = [x[0] for x in encoded_sequences]

        return (common_words, common_ids)
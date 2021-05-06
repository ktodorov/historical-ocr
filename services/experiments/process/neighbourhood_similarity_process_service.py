import os
from services.file_service import FileService
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.tagging_service import TaggingService
from services.log_service import LogService
from enums.part_of_speech import PartOfSpeech
from typing import Dict, List, Tuple


class NeighbourhoodSimilarityProcessService:
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            file_service: FileService,
            log_service: LogService,
            tagging_service: TaggingService):
        self._arguments_service = arguments_service
        self._file_service = file_service
        self._log_service = log_service
        self._tagging_service = tagging_service

    def get_target_tokens(
            self,
            cosine_distances: Dict[str, float],
            pos_tags: List[PartOfSpeech] = [PartOfSpeech.Noun, PartOfSpeech.Verb, PartOfSpeech.Adjective]) -> List[str]:
        metric_results = [(word, distance)
                          for word, distance in cosine_distances.items()
                          if self._tagging_service.word_is_specific_tag(word, pos_tags)]

        metric_results.sort(key=lambda x: x[1], reverse=True)

        most_changed_100 = [result[0] for result in metric_results[-100:]]
        most_changed_100_string = ', '.join(most_changed_100)
        self._log_service.log_debug(
            f'Most changed 100 words: [{most_changed_100_string}]')

        most_changed = self._map_target_tokens(
            metric_results,
            targets_count=10)

        log_message = f'Target words to be used: [' + \
            ', '.join(most_changed) + ']'
        self._log_service.log_info(log_message)

        return most_changed

    def _map_target_tokens(
            self,
            ordered_tuples: List[Tuple[str, float]],
            targets_count: int) -> List[str]:
        result_tuples = []
        preferred_tokens = self._get_preferred_target_tokens()

        for tuple in ordered_tuples:
            if preferred_tokens is None or tuple[0] in preferred_tokens:
                result_tuples.append(tuple[0])

            if len(result_tuples) == targets_count:
                return result_tuples

        return result_tuples

    def _get_preferred_target_tokens(self) -> List[str]:
        preferred_tokens_path = os.path.join(
            self._file_service.get_experiments_path(),
            f'preferred-tokens-{self._arguments_service.language.value}.txt')

        if not os.path.exists(preferred_tokens_path):
            return None

        preferred_tokens = []
        with open(preferred_tokens_path, 'r', encoding='utf-8') as tokens_file:
            file_lines = tokens_file.readlines()
            if file_lines is None or len(file_lines) == 0:
                return None

            preferred_tokens = [x.strip().lower() for x in file_lines]

        return preferred_tokens

from typing import Dict
from overrides import overrides
import argparse

from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService

from enums.ocr_output_type import OCROutputType


class OCRQualityNonContextArgumentsService(OCRQualityArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self, overwrite_args: Dict[str, object] = None) -> str:
        result = super().get_configuration_name(overwrite_args)

        rnd_value = self._get_value_or_default(overwrite_args, 'initialize_randomly', self.initialize_randomly)
        if rnd_value:
            result += f'-rnd'

        min_occurrence_value = self._get_value_or_default(overwrite_args, 'minimal_occurrence_limit', self.minimal_occurrence_limit)
        if min_occurrence_value is not None:
            result += f'-min{min_occurrence_value}'

        ocr_output_value = self._get_value_or_default(overwrite_args, 'ocr_output_type', self.ocr_output_type)
        output_type_suffix = ''
        if ocr_output_value == OCROutputType.GroundTruth:
            output_type_suffix = f'-grt'
        else:
            output_type_suffix = f'-{ocr_output_value.value}'

        result = result.replace(output_type_suffix, '')
        result += output_type_suffix

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--minimal-occurrence-limit', type=int, default=5,
                            help='Minimal occurrence limit for words or tokens to be included in the vocabulary. This setting is not taken into account for configurations using pre-trained vocabularies')

        parser.add_argument('--initialize-randomly', action='store_true',
                            help='If this is set to True, then the initial embeddings will be initialized randomly.')

        parser.add_argument('--window-size', type=int, default=5,
                            help='Window size to be used for models which rely on one such as CBOW and Skip-gram')

    @property
    def minimal_occurrence_limit(self) -> int:
        return self._get_argument('minimal_occurrence_limit')

    @property
    def initialize_randomly(self) -> bool:
        return self._get_argument('initialize_randomly')

    @property
    def window_size(self) -> int:
        return self._get_argument('window_size')
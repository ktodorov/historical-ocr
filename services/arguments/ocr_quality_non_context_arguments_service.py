from overrides import overrides
import argparse

from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService

from enums.ocr_output_type import OCROutputType


class OCRQualityNonContextArgumentsService(OCRQualityArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self) -> str:
        result = super().get_configuration_name()

        if self.initialize_randomly:
            result += f'-rnd'

        if self.separate_neighbourhood_vocabularies:
            result += f'-sep'

        if self.minimal_occurrence_limit is not None:
            result += f'-min{self.minimal_occurrence_limit}'

        if self.ocr_output_type == OCROutputType.GroundTruth:
            result += f'-grt'
        else:
            result += f'-{self.ocr_output_type.value}'

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--minimal-occurrence-limit', type=int, default=5,
                            help='Minimal occurrence limit for words or tokens to be included in the vocabulary. This setting is not taken into account for configurations using pre-trained vocabularies')

        parser.add_argument('--initialize-randomly', action='store_true',
                            help='If this is set to True, then the initial embeddings will be initialized randomly.')

    @property
    def minimal_occurrence_limit(self) -> int:
        return self._get_argument('minimal_occurrence_limit')

    @property
    def initialize_randomly(self) -> bool:
        return self._get_argument('initialize_randomly')
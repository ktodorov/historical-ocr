from overrides import overrides
import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.ocr_output_type import OCROutputType


class OCREvaluationArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self) -> str:
        result = super().get_configuration_name()

        if self.initialize_randomly:
            result += f'-rnd'

        if self.minimal_occurrence_limit is not None:
            result += f'-min{self.minimal_occurrence_limit}'

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--minimal-occurrence-limit', type=int, default=None,
                            help='Minimal occurrence limit for words or tokens to be included in the vocabulary. This setting is not taken into account for configurations using pre-trained vocabularies')

        parser.add_argument('--separate-neighbourhood-vocabularies', action='store_true',
                            help='If this is set to True, then the neighbourhood similarity graph will use separate vocabularies of the models')

        parser.add_argument('--initialize-randomly', action='store_true',
                            help='If this is set to True, then the initial embeddings will be initialized randomly.')

        parser.add_argument('--neighbourhood-set-size', type=int, default=1000,
                            help='The neighbourhood_set_size set size. Larger values tend to produce more stable results. Default value is 1000.')

    @property
    def minimal_occurrence_limit(self) -> int:
        return self._get_argument('minimal_occurrence_limit')

    @property
    def separate_neighbourhood_vocabularies(self) -> bool:
        return self._get_argument('separate_neighbourhood_vocabularies')

    @property
    def initialize_randomly(self) -> bool:
        return self._get_argument('initialize_randomly')

    @property
    def neighbourhood_set_size(self) -> int:
        return self._get_argument('neighbourhood_set_size')
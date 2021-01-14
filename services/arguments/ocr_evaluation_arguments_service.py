from overrides import overrides
import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.ocr_output_type import OCROutputType


class OCREvaluationArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self) -> str:
        result = f'{str(self.language)[:2]}'
        result += f'-{self.configuration.value}'

        if self.initialize_randomly:
            result += f'-rnd'

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--minimal-occurrence-limit', type=int, default=5,
                            help='Minimal occurrence limit for words or tokens to be included in the vocabulary. This setting is not taken into account for configurations using pre-trained vocabularies')

        parser.add_argument('--separate-neighbourhood-vocabularies', action='store_true',
                            help='If this is set to True, then the neighbourhood similarity graph will use separate vocabularies of the models')

        parser.add_argument('--initialize-randomly', action='store_true',
                            help='If this is set to True, then the initial embeddings will be initialized randomly.')

    @property
    def minimal_occurrence_limit(self) -> int:
        return self._get_argument('minimal_occurrence_limit')

    @property
    def separate_neighbourhood_vocabularies(self) -> bool:
        return self._get_argument('separate_neighbourhood_vocabularies')

    @property
    def initialize_randomly(self) -> bool:
        return self._get_argument('initialize_randomly')
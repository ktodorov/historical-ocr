from overrides import overrides
import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.ocr_output_type import OCROutputType


class OCRQualityArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    @overrides
    def get_configuration_name(self) -> str:
        result = f'{str(self.language)[:2]}'
        result += f'-{self.configuration.value}'

        if self.ocr_output_type == OCROutputType.GroundTruth:
            result += f'-grt'
        else:
            result += f'-{self.ocr_output_type.value}'

        return result

    @overrides
    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--ocr-output-type', type=OCROutputType, choices=list(OCROutputType), required=True,
                            help='OCR output type to be used')

        parser.add_argument('--minimal-occurrence-limit', type=int, default=5,
                            help='Minimal occurrence limit for words or tokens to be included in the vocabulary. This setting is not taken into account for configurations using pre-trained vocabularies')

    @property
    def ocr_output_type(self) -> OCROutputType:
        return self._get_argument('ocr_output_type')

    @property
    def minimal_occurrence_limit(self) -> int:
        return self._get_argument('minimal_occurrence_limit')

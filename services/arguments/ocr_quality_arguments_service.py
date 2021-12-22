from typing import Dict
from overrides import overrides
import argparse

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService

from enums.ocr_output_type import OCROutputType


class OCRQualityArgumentsService(PretrainedArgumentsService):
    def __init__(self):
        super().__init__()

    def get_configuration_name(self, overwrite_args: Dict[str, object] = None) -> str:
        result = super().get_configuration_name(overwrite_args)

        ocr_output_value = self._get_value_or_default(overwrite_args, 'ocr_output_type', self.ocr_output_type)
        output_type_suffix = ''
        if ocr_output_value == OCROutputType.GroundTruth:
            output_type_suffix = f'-grt'
        else:
            output_type_suffix = f'-{ocr_output_value.value}'

        result = result.replace(output_type_suffix, '')
        result += output_type_suffix

        return result

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        super()._add_specific_arguments(parser)

        parser.add_argument('--ocr-output-type', type=OCROutputType, choices=list(OCROutputType), required=True,
                            help='OCR output type to be used')

    @property
    def ocr_output_type(self) -> OCROutputType:
        return self._get_argument('ocr_output_type')

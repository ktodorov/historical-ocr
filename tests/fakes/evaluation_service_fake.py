from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from overrides import overrides
from tests.utils.argument_utils import default_values

class EvaluationServiceFake(OCREvaluationArgumentsService):
    def __init__(self, custom_values = {}):
        super().__init__()

        self._custom_values = custom_values

    @overrides
    def _parse_arguments(self):
        return

    @overrides
    def _get_argument(self, key: str) -> object:
        if key not in self._custom_values.keys():
            return default_values[key]

        return self._custom_values[key]

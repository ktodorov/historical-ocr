from overrides import overrides

from services.arguments.arguments_service_base import ArgumentsServiceBase

class ArgumentServiceFake(ArgumentsServiceBase):
    def __init__(self, custom_values = {}):
        super().__init__(raise_errors_on_invalid_args=True)

        self._custom_values = custom_values

    @overrides
    def _parse_arguments(self):
        return

    @overrides
    def _get_argument(self, key: str) -> object:
        if key not in self._custom_values.keys():
            return None

        return self._custom_values[key]

from overrides import overrides

from enums.language import Language

from models.model_base import ModelBase

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.log_service import LogService


class ModelFake(ModelBase):
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            log_service: LogService):
        super().__init__(data_service, arguments_service, log_service)

    @overrides
    def forward(self, input_batch, **kwargs):
        return None

    def _get_embedding_size(self, language: Language):
        if language == Language.English:
            return 300
        elif language == Language.Dutch:
            return 320

        raise NotImplementedError()
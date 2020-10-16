from enums.configuration import Configuration

from models.model_base import ModelBase

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.data_service import DataService
from services.vocabulary_service import VocabularyService
from services.process.process_service_base import ProcessServiceBase
from services.file_service import FileService

class ModelService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            data_service: DataService,
            vocabulary_service: VocabularyService,
            process_service: ProcessServiceBase,
            file_service: FileService):
        self._arguments_service = arguments_service
        self._data_service = data_service
        self._vocabulary_service = vocabulary_service
        self._process_service = process_service
        self._file_service = file_service

    def create_model(self) -> ModelBase:
        configuration: Configuration = self._arguments_service.configuration

        device = self._arguments_service.device

        pass # TODO

        raise LookupError(f'The {str(configuration)} is not supported')

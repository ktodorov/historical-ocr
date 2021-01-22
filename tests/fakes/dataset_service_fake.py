from tests.fakes.skip_gram_dataset_fake import SkipGramDatasetFake
from datasets.dataset_base import DatasetBase
from enums.run_type import RunType

from services.cache_service import CacheService
from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.log_service import LogService
from services.process.process_service_base import ProcessServiceBase


class DatasetServiceFake:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            cache_service: CacheService,
            process_service: ProcessServiceBase,
            log_service: LogService):

        self._arguments_service = arguments_service
        self._cache_service = cache_service
        self._process_service = process_service
        self._log_service = log_service

    def initialize_dataset(self, run_type: RunType) -> DatasetBase:
        return SkipGramDatasetFake(
            arguments_service=self._arguments_service,
            process_service=self._process_service,
            log_service=self._log_service,
            cache_service=self._cache_service)
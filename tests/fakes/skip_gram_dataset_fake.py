from overrides import overrides

from datasets.skip_gram_dataset import SkipGramDataset
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService
from services.log_service import LogService
from services.cache_service import CacheService
from services.process.skip_gram_process_service import SkipGramProcessService


class SkipGramDatasetFake(SkipGramDataset):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: SkipGramProcessService,
            log_service: LogService,
            cache_service: CacheService):
        super().__init__(arguments_service, process_service, log_service)

        self._ids = []

    @overrides
    def __getitem__(self, idx):
        document_ids = [x.document_index for x in self._text_corpus.get_entries(idx)]
        self._ids.extend(document_ids)

        return super().__getitem__(idx)
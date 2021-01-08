from services.log_service import LogService
from entities.tokens_occurrence_stats import TokensOccurrenceStats
from overrides import overrides

from datasets.dataset_base import DatasetBase
from services.arguments.ocr_quality_arguments_service import OCRQualityArgumentsService

from services.process.ppmi_process_service import PPMIProcessService

class PPMIDataset(DatasetBase):
    def __init__(
            self,
            arguments_service: OCRQualityArgumentsService,
            process_service: PPMIProcessService,
            log_service: LogService,
            **kwargs):
        super().__init__()

        self._arguments_service = arguments_service
        self._log_service = log_service

        self._occurrence_stats: TokensOccurrenceStats = process_service.get_occurrence_stats(ocr_output_type=self._arguments_service.ocr_output_type)
        self._log_service.log_debug(f'Loaded occurrence matrix with shape {self._occurrence_stats.mutual_occurrences.shape}')

    @overrides
    def __len__(self):
        return 1 # TODO Check

    @overrides
    def __getitem__(self, idx):
        return self._occurrence_stats

    @overrides
    def use_collate_function(self) -> bool:
        return True

    @overrides
    def collate_function(self, sequences):
        stats = sequences[0]
        return stats

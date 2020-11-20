import math
from overrides.overrides import overrides
from typing import Counter, List, Dict
from overrides import overrides
from scipy.spatial import procrustes

from decimal import Decimal

from enums.experiment_type import ExperimentType

from models.model_base import ModelBase

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.dataloader_service import DataLoaderService
from services.experiments.experiment_service_base import ExperimentServiceBase
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.plot_service import PlotService


class OCRQualityExperimentService(ExperimentServiceBase):
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            dataloader_service: DataLoaderService,
            file_service: FileService,
            metrics_service: MetricsService,
            plot_service: PlotService,
            model: ModelBase):
        super().__init__(arguments_service, dataloader_service, file_service, model)

        self._metrics_service = metrics_service
        self._plot_service = plot_service

    @overrides
    def execute_experiments(self, experiment_types: List[ExperimentType]):

        result = {experiment_type: {} for experiment_type in experiment_types}

        model1_embeddings = []
        model2_embeddings = []

        dataloader_length = len(self._dataloader)
        for i, batch in enumerate(self._dataloader):
            print(f'{i}/{dataloader_length}         \r', end='')

            words, token_ids = batch
            outputs = self._model.get_embeddings(token_ids)

            model1_embeddings.extend(outputs[0])
            model2_embeddings.extend(outputs[1])

            for i, word in enumerate(words):
                if ExperimentType.CosineDistance in experiment_types:
                    result[ExperimentType.CosineDistance][word] = self._metrics_service.calculate_cosine_distance(
                        list1=outputs[0][i],
                        list2=outputs[1][i])

                if ExperimentType.EuclideanDistance in experiment_types:
                    result[ExperimentType.EuclideanDistance][word] = self._metrics_service.calculate_euclidean_distance(
                        list1=outputs[0][i],
                        list2=outputs[1][i])

                # if ExperimentType.KLDivergence in experiment_types:
                #     result[ExperimentType.KLDivergence][word] = self._metrics_service.calculate_KL_divergence(
                #         list1=outputs[0][i],
                #         list2=outputs[1][i])

        # a, b, c = procrustes(model1_embeddings, model2_embeddings)

        self._save_experiment_results(result)

    def _save_experiment_results(self, result: Dict[ExperimentType, Dict[str, float]]):
        experiments_folder = self._file_service.get_experiments_path()
        distances_folder = self._file_service.combine_path(
            experiments_folder, 'distances', create_if_missing=True)

        for experiment_type, word_value_pairs in result.items():
            values = list(word_value_pairs.values())
            
            if values is None or len(values) == 0:
                continue

            exponents = [Decimal(str(x)).as_tuple().exponent for x in values]
            counter = Counter(exponents)
            counter = Counter(
                {f'e{str(key)}': value for key, value in counter.items()})

            self._plot_service.plot_counters_histogram(
                counter_labels=['a'],
                counters=[counter],
                title=experiment_type.value,
                show_legend=False,
                plot_values_above_bars=True,
                save_path=distances_folder,
                filename=experiment_type.value)

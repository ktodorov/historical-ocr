from services.plots.ocr_neighbour_overlap_plot_service import OCRNeighbourOverlapPlotService
from services.embeddings.word_embeddings_service import WordEmbeddingsService
from enums.configuration import Configuration
from enums.overlap_type import OverlapType

from services.experiments.process.metrics_process_service import MetricsProcessService


from entities.cache.cache_options import CacheOptions
from entities.word_evaluation import WordEvaluation
from services.cache_service import CacheService
from overrides.overrides import overrides
from typing import Callable, List, Dict
from overrides import overrides

from enums.experiment_type import ExperimentType

from models.model_base import ModelBase

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.dataloader_service import DataLoaderService
from services.experiments.experiment_service_base import ExperimentServiceBase
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.experiments.process.word_neighbourhood_service import WordNeighbourhoodService
from services.log_service import LogService

from services.plots.baseline_neighbour_overlap_plot_service import BaselineNeighbourOverlapPlotService
from services.plots.individual_metrics_plot_service import IndividualMetricsPlotService
from services.plots.set_sized_based_plot_service import SetSizedBasedPlotService


class OCRQualityExperimentService(ExperimentServiceBase):
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            dataloader_service: DataLoaderService,
            file_service: FileService,
            metrics_service: MetricsService,
            cache_service: CacheService,
            word_neighbourhood_service: WordNeighbourhoodService,
            log_service: LogService,
            metrics_process_service: MetricsProcessService,
            baseline_neighbour_overlap_plot_service: BaselineNeighbourOverlapPlotService,
            ocr_neighbour_overlap_plot_service: OCRNeighbourOverlapPlotService,
            individual_metrics_plot_service: IndividualMetricsPlotService,
            set_sized_based_plot_service: SetSizedBasedPlotService,
            word_embeddings_service: WordEmbeddingsService,
            model: ModelBase):
        super().__init__(arguments_service, dataloader_service, file_service, model)

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._cache_service = cache_service
        self._word_neighbourhood_service = word_neighbourhood_service
        self._log_service = log_service
        self._metrics_process_service = metrics_process_service
        self._word_embeddings_service = word_embeddings_service

        # plot services
        self._baseline_neighbour_overlap_plot_service = baseline_neighbour_overlap_plot_service
        self._ocr_neighbour_overlap_plot_service = ocr_neighbour_overlap_plot_service
        self._individual_metrics_plot_service = individual_metrics_plot_service
        self._set_sized_based_plot_service = set_sized_based_plot_service

        self._random_suffix = '-rnd' if self._arguments_service.initialize_randomly else ''
        self._separate_suffix = '-sep' if self._arguments_service.separate_neighbourhood_vocabularies else ''
        self._lr_suffix = f'-lr{self._arguments_service.get_learning_rate_str()}' if self._arguments_service.configuration != Configuration.PPMI else ''

    @overrides
    def execute_experiments(self, experiment_types: List[ExperimentType]):
        experiment_types_str = ', '.join([x.value for x in experiment_types])
        self._log_service.log_debug(
            f'Executing experiments: {experiment_types_str}')

        result = {experiment_type: {} for experiment_type in experiment_types}

        # Load word evaluations
        word_evaluations = self._load_word_evaluations()

        # Cosine distances
        self._load_experiment_result(ExperimentType.CosineDistance, experiment_types,
                                     result, lambda: self._load_cosine_distances(word_evaluations))

        # Neighbourhood overlaps
        self._load_experiment_result(ExperimentType.NeighbourhoodOverlap, experiment_types,
                                     result, lambda: self._load_neighbourhood_overlaps(word_evaluations))

        # Neighbourhood plots
        if ExperimentType.CosineDistance in experiment_types and ExperimentType.NeighbourhoodOverlap in experiment_types:
            self._word_neighbourhood_service.generate_neighbourhood_plots(
                word_evaluations,
                result[ExperimentType.CosineDistance])

        # # Euclidean distances
        # self._load_experiment_result(ExperimentType.EuclideanDistance, experiment_types,
        #                              result, lambda: self._load_euclidean_distances(word_evaluations))

        # Save final results and generate plots
        self._save_experiment_results(result, word_evaluations)

        self._log_service.log_info(
            'Experiments calculation completed successfully')

    def _load_word_evaluations(self):
        word_evaluations: List[WordEvaluation] = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'word-evaluations',
                key_suffixes=[
                    self._random_suffix,
                    self._separate_suffix,
                    self._lr_suffix
                ],
                seed_specific=True),
            callback_function=lambda: self._word_embeddings_service.generate_embeddings(self._model, self._dataloader))

        self._log_service.log_info('Loaded word evaluations')

        return word_evaluations

    def _load_experiment_result(
            self,
            experiment_type: ExperimentType,
            experiment_types: List[ExperimentType],
            result_dict: Dict[ExperimentType, dict],
            callback_func: Callable):
        if experiment_type not in experiment_types:
            return

        result_dict[experiment_type] = callback_func()

    def _load_cosine_distances(self, word_evaluations: List[WordEvaluation]):
        result = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'cosine-distances',
                key_suffixes=[
                    self._random_suffix,
                    self._lr_suffix
                ],
                seed_specific=True),
            callback_function=lambda: self._metrics_process_service.calculate_cosine_distances(word_evaluations))

        self._log_service.log_info('Loaded cosine distances')
        return result

    def _load_neighbourhood_overlaps(self, word_evaluations: List[WordEvaluation]):
        result = {}
        for overlap_type in OverlapType:
            if ((self._arguments_service.initialize_randomly and overlap_type != OverlapType.GTvsOCR) or
                (self._arguments_service.configuration == Configuration.PPMI and overlap_type == OverlapType.BASEvsOG) or 
                (overlap_type == OverlapType.BASEvsOG)):
                continue

            result[overlap_type] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    'neighbourhood-overlaps',
                    key_suffixes=[
                        self._lr_suffix,
                        '-',
                        overlap_type.value,
                        self._random_suffix
                    ],
                    seed_specific=True),
                callback_function=lambda: self._word_neighbourhood_service.generate_neighbourhood_similarities(
                    word_evaluations,
                    overlap_type=overlap_type))

        self._log_service.log_info('Loaded neighbourhood overlaps')
        return result

    def _load_euclidean_distances(self, word_evaluations: List[WordEvaluation]):
        result = self._cache_service.get_item_from_cache(
            CacheOptions(
                'euclidean-distances',
                key_suffixes=[
                    self._random_suffix,
                    self._lr_suffix
                ],
                seed_specific=True),
            callback_function=lambda: self._metrics_process_service.calculate_euclidean_distances(word_evaluations))

        self._log_service.log_info('Loaded euclidean distances')
        return result

    def _save_experiment_results(self, result: Dict[ExperimentType, Dict[str, float]], word_evaluations: List[WordEvaluation]):
        self._log_service.log_info('Saving experiment results')

        # Plot individual metrics
        # self._individual_metrics_plot_service.plot_individual_metrics(result)

        # Baseline vs. others plots
        # self._baseline_neighbour_overlap_plot_service.plot_baseline_overlaps()

        # GT vs OCR plots
        total_words_count = len(
            [1 for x in word_evaluations if x.contains_all_embeddings(OverlapType.GTvsOCR)])
        self._ocr_neighbour_overlap_plot_service.plot_ocr_ground_truth_overlaps(
            total_words_count)

        # Set size based plots
        # self._set_sized_based_plot_service.plot_set_size_bases()

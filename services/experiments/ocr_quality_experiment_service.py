from enums.configuration import Configuration
from enums.value_summary import ValueSummary
from entities.plot.legend_options import LegendOptions
from enums.overlap_type import OverlapType
import numpy as np

from scipy.spatial import procrustes
from entities.plot.plot_options import PlotOptions
from services.experiments.process.metrics_process_service import MetricsProcessService
from services.experiments.process.neighbourhood_similarity_process_service import NeighbourhoodSimilarityProcessService
from services.experiments.process.neighbourhood_overlap_process_service import NeighbourhoodOverlapProcessService

from entities.plot.figure_options import FigureOptions

from entities.cache.cache_options import CacheOptions
from services.tagging_service import TaggingService
from enums.ocr_output_type import OCROutputType
from services.vocabulary_service import VocabularyService
from entities.word_evaluation import WordEvaluation
from services.cache_service import CacheService
from overrides.overrides import overrides
from typing import Counter, List, Dict
from overrides import overrides
from tqdm import tqdm

from enums.experiment_type import ExperimentType

from models.model_base import ModelBase

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.dataloader_service import DataLoaderService
from services.experiments.experiment_service_base import ExperimentServiceBase
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.plot_service import PlotService
from services.experiments.process.word_neighbourhood_service import WordNeighbourhoodService
from services.log_service import LogService


class OCRQualityExperimentService(ExperimentServiceBase):
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            dataloader_service: DataLoaderService,
            file_service: FileService,
            metrics_service: MetricsService,
            plot_service: PlotService,
            cache_service: CacheService,
            word_neighbourhood_service: WordNeighbourhoodService,
            vocabulary_service: VocabularyService,
            log_service: LogService,
            tagging_service: TaggingService,
            neighbourhood_overlap_process_service: NeighbourhoodOverlapProcessService,
            metrics_process_service: MetricsProcessService,
            model: ModelBase):
        super().__init__(arguments_service, dataloader_service, file_service, model)

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._cache_service = cache_service
        self._word_neighbourhood_service = word_neighbourhood_service
        self._vocabulary_service = vocabulary_service
        self._log_service = log_service
        self._tagging_service = tagging_service
        self._neighbourhood_overlap_process_service = neighbourhood_overlap_process_service
        self._metrics_process_service = metrics_process_service

    @overrides
    def execute_experiments(self, experiment_types: List[ExperimentType]):
        experiment_types_str = ', '.join([x.value for x in experiment_types])
        self._log_service.log_debug(
            f'Executing experiments: {experiment_types_str}')

        result = {experiment_type: {} for experiment_type in experiment_types}

        random_suffix = '-rnd' if self._arguments_service.initialize_randomly else ''
        separate_suffix = '-sep' if self._arguments_service.separate_neighbourhood_vocabularies else ''
        lr_suffix = f'-lr{self._arguments_service.get_learning_rate_str()}' if self._arguments_service.configuration != Configuration.PPMI else ''
        word_evaluations: List[WordEvaluation] = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'word-evaluations',
                key_suffixes=[
                    random_suffix,
                    separate_suffix,
                    lr_suffix
                ],
                seed_specific=True),
            callback_function=self._generate_embeddings)

        self._log_service.log_info('Loaded word evaluations')

        if ExperimentType.CosineDistance in experiment_types:
            result[ExperimentType.CosineDistance] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'cosine-distances',
                    key_suffixes=[
                        random_suffix,
                        lr_suffix
                    ],
                    seed_specific=True),
                callback_function=lambda: self._metrics_process_service.calculate_cosine_distances(word_evaluations))

            self._log_service.log_info('Loaded cosine distances')

        if ExperimentType.NeighbourhoodOverlap in experiment_types:
            result[ExperimentType.NeighbourhoodOverlap] = {}
            for overlap_type in OverlapType:
                if overlap_type == OverlapType.GTvsBase and self._arguments_service.initialize_randomly:
                    continue

                result[ExperimentType.NeighbourhoodOverlap][overlap_type] = self._cache_service.get_item_from_cache(
                    CacheOptions(
                        'neighbourhood-overlaps',
                        key_suffixes=[
                            lr_suffix,
                            '-',
                            overlap_type.value,
                            random_suffix,
                            f'-{self._arguments_service.neighbourhood_set_size}'
                        ],
                        seed_specific=True),
                    callback_function=lambda: self._word_neighbourhood_service.generate_neighbourhood_similarities(
                        word_evaluations,
                        neighbourhood_set_size=self._arguments_service.neighbourhood_set_size,
                        overlap_type=overlap_type))

            self._log_service.log_info('Loaded neighbourhood overlaps')

        if ExperimentType.CosineDistance in experiment_types and ExperimentType.NeighbourhoodOverlap in experiment_types:
            self._word_neighbourhood_service.generate_neighbourhood_plots(
                word_evaluations,
                result[ExperimentType.CosineDistance])

        if ExperimentType.EuclideanDistance in experiment_types:
            result[ExperimentType.EuclideanDistance] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'euclidean-distances{random_suffix}',
                    seed_specific=True),
                callback_function=lambda: self._metrics_process_service.calculate_euclidean_distances(word_evaluations))

            self._log_service.log_info('Loaded euclidean distances')

        self._log_service.log_info('Saving experiment results')
        self._save_experiment_results(result)

        self._log_service.log_info(
            'Experiments calculation completed successfully')

    def _generate_embeddings(self) -> List[WordEvaluation]:
        result: List[WordEvaluation] = []

        self._log_service.log_debug('Processing common vocabulary tokens')
        dataloader_length = len(self._dataloader)
        for i, tokens in tqdm(iterable=enumerate(self._dataloader), desc=f'Processing common vocabulary tokens', total=dataloader_length):
            # tokens = batch
            outputs = self._model.get_embeddings(tokens)
            result.extend(outputs)

        if self._arguments_service.initialize_randomly:
            result = self._align_word_embeddings(result)
        elif self._arguments_service.separate_neighbourhood_vocabularies:
            processed_tokens = [we.word for we in result]
            for ocr_output_type in [OCROutputType.Raw, OCROutputType.GroundTruth]:
                self._log_service.log_debug(
                    f'Processing unique vocabulary tokens for {ocr_output_type.value} type')
                vocab_key = f'vocab-{self._arguments_service.get_dataset_string()}-{ocr_output_type.value}'
                self._vocabulary_service.load_cached_vocabulary(vocab_key)
                unprocessed_tokens = []
                for _, token in self._vocabulary_service.get_vocabulary_tokens(exclude_special_tokens=True):
                    if token in processed_tokens:
                        continue

                    unprocessed_tokens.append(token)

                batch_size = self._arguments_service.batch_size
                with tqdm(desc=f'Unique {ocr_output_type.value} vocabulary tokens', total=len(unprocessed_tokens)) as progress_bar:
                    for i in range(0, len(unprocessed_tokens), batch_size):
                        tokens = unprocessed_tokens[i:i+batch_size]
                        word_evaluations = self._model.get_embeddings(
                            tokens, skip_unknown=True)
                        result.extend(word_evaluations)
                        processed_tokens.extend(tokens)
                        progress_bar.update(len(tokens))

        return result

    def _save_experiment_results(self, result: Dict[ExperimentType, Dict[str, float]]):
        self._save_individual_experiments(result)
        self._log_service.log_info('Generating plots')
        self._generate_common_plots()
        # self._generate_set_sized_based_plot()

    def _save_individual_experiments(self, result: Dict[ExperimentType, Dict[str, float]]):

        for experiment_type, word_value_pairs_by_overlap in result.items():
            if experiment_type == ExperimentType.NeighbourhoodOverlap:
                for overlap_type, word_value_pairs in word_value_pairs_by_overlap.items():
                    self._save_individual_experiment(experiment_type, overlap_type, word_value_pairs)
            else:
                self._save_individual_experiment(experiment_type, None, word_value_pairs_by_overlap)

    def _save_individual_experiment(
        self,
        experiment_type: ExperimentType,
        overlap_type: OverlapType,
        word_value_pairs):
        experiments_folder = self._file_service.get_experiments_path()
        experiment_type_folder = self._file_service.combine_path(
            experiments_folder,
            experiment_type.value,
            create_if_missing=True)

        if overlap_type is not None:
            experiment_type_folder = self._file_service.combine_path(
                experiment_type_folder,
                overlap_type.value,
                create_if_missing=True)

        self._log_service.log_debug(
            f'Saving \'{experiment_type.value}\' experiment results at \'{experiment_type_folder}\'')

        values = [round(x, 1) for x in word_value_pairs.values()]
        if values is None or len(values) == 0:
            return

        counter = Counter(values)
        filename = f'{self._arguments_service.get_configuration_name()}-{self._arguments_service.neighbourhood_set_size}'
        self._plot_service.plot_distribution(
            counts=counter,
            plot_options=PlotOptions(
                legend_options=LegendOptions(show_legend=False),
                figure_options=FigureOptions(
                    title=experiment_type.value,
                    save_path=experiment_type_folder,
                    filename=filename),
                color='royalblue',
                fill=True))

    def _generate_common_plots(self):
        experiments_folder = self._file_service.get_experiments_path()
        experiment_type_folder = self._file_service.combine_path(
            experiments_folder,
            ExperimentType.NeighbourhoodOverlap.value,
            self._arguments_service.language.value,
            create_if_missing=True)

        # main_ax = self._plot_service.create_plot()
        overlaps_by_config = self._neighbourhood_overlap_process_service.get_overlaps(
            self._arguments_service.neighbourhood_set_size)
        fig, config_axs = self._plot_service.create_plots(len(overlaps_by_config.keys()))

        for i, (configuration, overlaps_by_lr) in enumerate(overlaps_by_config.items()):
            for learning_rate, overlaps_by_type in overlaps_by_lr.items():
                for overlap_type, overlaps_by_seed in overlaps_by_type.items():
                    if all(x is None for x in list(overlaps_by_seed.values())):
                        continue

                    combined_overlaps = self._neighbourhood_overlap_process_service.combine_seed_overlaps(
                        overlaps_by_seed,
                        self._arguments_service.neighbourhood_set_size)

                    value_summaries = self._neighbourhood_overlap_process_service.extract_value_summaries(
                        combined_overlaps)

                    for value_summary, overlap_line in value_summaries.items():
                        # Skip max and min value summaries
                        if value_summary != ValueSummary.Average:
                            continue

                        config_axs[i] = self._plot_service.plot_distribution(
                            counts=overlap_line,
                            plot_options=self._neighbourhood_overlap_process_service.get_distribution_plot_options(
                                config_axs[i],
                                configuration,
                                overlap_type,
                                learning_rate,
                                value_summary))

            #             config_ax = self._plot_service.plot_distribution(
            #                 counts=overlap_line,
            #                 plot_options=self._neighbourhood_overlap_process_service.get_distribution_plot_options(
            #                     config_ax,
            #                     configuration,
            #                     overlap_type,
            #                     learning_rate,
            #                     value_summary))
            self._plot_service.set_plot_properties(
                ax=config_axs[i],
                figure_options=FigureOptions(
                    hide_y_labels=True,
                    figure=fig,
                    super_title=f'Neighbourhood overlaps ({self._arguments_service.language.value})',
                    title=str(configuration.value)))

            # self._plot_service.save_plot(
            #     save_path=experiment_type_folder,
            #     filename=f'combined-neighbourhood-overlaps-{self._arguments_service.configuration.value}-{self._arguments_service.neighbourhood_set_size}')

        # self._plot_service.set_plot_properties(
        #     ax=main_ax,
        #     figure_options=FigureOptions(
        #         title=f'Neighbourhood overlaps ({self._arguments_service.language.value})'))

        self._plot_service.save_plot(
            save_path=experiment_type_folder,
            filename=f'neighbourhood-overlaps-{self._arguments_service.neighbourhood_set_size}',
            figure=fig)

    def _align_word_embeddings(self, evaluations: List[WordEvaluation]) -> List[WordEvaluation]:
        if len(evaluations) == 0:
            raise Exception('Evaluations list is empty')

        embeddings_size = evaluations[0].get_embeddings_size()
        model1_embeddings = np.zeros((len(evaluations), embeddings_size))
        model2_embeddings = np.zeros((len(evaluations), embeddings_size))

        for i, word_evaluation in enumerate(evaluations):
            model1_embeddings[i] = word_evaluation.get_embeddings(0)
            model2_embeddings[i] = word_evaluation.get_embeddings(1)

        standardized_model1_embeddings, standardized_model2_embeddings, disparity = procrustes(
            model1_embeddings, model2_embeddings)
        self._log_service.log_debug(f'Disparity found: {disparity}')

        new_evaluations = []
        for i, word_evaluation in enumerate(evaluations):
            new_evaluations.append(WordEvaluation(
                word_evaluation.word,
                [standardized_model1_embeddings[i],
                 standardized_model2_embeddings[i]]))

        return new_evaluations

    def _generate_set_sized_based_plot(self):
        experiments_folder = self._file_service.get_experiments_path()
        experiment_type_folder = self._file_service.combine_path(
            experiments_folder,
            ExperimentType.OverlapSetSizeComparison.value,
            create_if_missing=True)

        ax = self._plot_service.create_plot(
            PlotOptions(
                figure_options=FigureOptions(
                    seaborn_style='whitegrid')))
        percentages = {}

        for set_size in range(100, 1050, 50):
            overlaps_by_config_and_seed = self._neighbourhood_overlap_process_service.get_calculated_overlaps(
                set_size)

            for (configuration, is_random_initialized), overlaps_by_seed in overlaps_by_config_and_seed.items():
                if (configuration, is_random_initialized) not in percentages.keys():
                    percentages[(configuration, is_random_initialized)] = {}

                if all(x is None for x in list(overlaps_by_seed.values())):
                    continue

                combined_overlaps = self._neighbourhood_overlap_process_service.combine_seed_overlaps(
                    overlaps_by_seed,
                    set_size,
                    max_bins=set_size)

                percentage_value = self._get_overlap_percentage(
                    combined_overlaps, set_size)
                percentages[(configuration, is_random_initialized)
                            ][set_size] = percentage_value

        keys_to_delete = [
            key for key, values in percentages.items() if all(x == 0 for x in values)]
        for key in keys_to_delete:
            del percentages[key]

        set_sizes = [x for x in range(100, 1050, 50)]
        y_values = [[0 for _ in range(len(set_sizes))]
                    for _ in range(len(percentages.keys()))]
        for i, ((configuration, is_random_initialized), percentages_by_set_size) in enumerate(percentages.items()):
            for k, set_size in enumerate(set_sizes):
                if set_size in percentages_by_set_size.keys():
                    y_values[i][k] = percentages_by_set_size[set_size]

        labels = [
            f'{configuration.value}' +
            ('[random]' if is_random_initialized else '')
            for (configuration, is_random_initialized) in percentages.keys()]

        self._plot_service.plot_lines(
            x_values=set_sizes,
            y_values=y_values,
            labels=labels,
            plot_options=PlotOptions(
                ax=ax,
                figure_options=FigureOptions(
                    title=f'{ExperimentType.OverlapSetSizeComparison.value}-{self._arguments_service.language.value}',
                    save_path=experiment_type_folder,
                    filename=f'overlap-set-size-comparison-{self._arguments_service.language.value}')))

    def _get_overlap_percentage(self, combined_overlaps: Dict[int, List[int]], set_size: int):
        result = 0
        count = 0

        for overlap_amount, overlap_values in combined_overlaps.items():
            current_percentage = overlap_amount / set_size
            for overlap_value in overlap_values:
                if overlap_value is None:
                    continue

                result = result + (overlap_value * current_percentage)
                count = count + overlap_value

        final_percentage = result / count
        return final_percentage

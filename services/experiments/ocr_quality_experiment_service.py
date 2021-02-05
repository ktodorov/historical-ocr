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
from typing import Counter, List, Dict, Tuple
from overrides import overrides

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
        word_evaluations: List[WordEvaluation] = self._cache_service.get_item_from_cache(
            CacheOptions(
                f'word-evaluations{random_suffix}{separate_suffix}',
                seed_specific=True),
            callback_function=self._generate_embeddings)

        self._log_service.log_info('Loaded word evaluations')

        # if ExperimentType.CosineSimilarity in experiment_types:
        #     result[ExperimentType.CosineSimilarity] = self._cache_service.get_item_from_cache(
        #         item_key=f'cosine-similarities{random_suffix}',
        #         callback_function=lambda: self._metrics_process_service.calculate_cosine_similarities(word_evaluations))
        #     self._log_service.log_info('Loaded cosine similarities')

        if ExperimentType.CosineDistance in experiment_types:
            result[ExperimentType.CosineDistance] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'cosine-distances{random_suffix}',
                    seed_specific=True),
                callback_function=lambda: self._metrics_process_service.calculate_cosine_distances(word_evaluations))

            self._log_service.log_info('Loaded cosine distances')

        if ExperimentType.NeighbourhoodOverlap in experiment_types:
            result[ExperimentType.NeighbourhoodOverlap] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'neighbourhood-overlaps{random_suffix}',
                    seed_specific=True),
                callback_function=lambda: self._word_neighbourhood_service.generate_neighbourhood_similarities(word_evaluations))

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

        # a, b, c = procrustes(model1_embeddings, model2_embeddings)

        self._log_service.log_info('Saving experiment results')
        self._save_experiment_results(result)

        self._log_service.log_info(
            'Experiments calculation completed successfully')

    def _generate_embeddings(self) -> List[WordEvaluation]:
        result: List[WordEvaluation] = []

        self._log_service.log_debug('Processing common vocabulary tokens')
        dataloader_length = len(self._dataloader)
        for i, batch in tqdm(iterable=enumerate(self._dataloader), desc=f'Processing common vocabulary tokens', total=dataloader_length):
            tokens, vocab_ids = batch
            outputs = self._model.get_embeddings(tokens, vocab_ids)
            result.extend(outputs)

        if self._arguments_service.separate_neighbourhood_vocabularies:
            processed_tokens = [we.word for we in result]
            for ocr_output_type in [OCROutputType.Raw, OCROutputType.GroundTruth]:
                self._log_service.log_debug(
                    f'Processing unique vocabulary tokens for {ocr_output_type.value} type')
                vocab_key = f'vocab-{ocr_output_type.value}'
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
                            tokens, vocab_ids=None, skip_unknown=True)
                        result.extend(word_evaluations)
                        processed_tokens.extend(tokens)
                        progress_bar.update(len(tokens))

        return result

    def _save_experiment_results(self, result: Dict[ExperimentType, Dict[str, float]]):
        self._save_individual_experiments(result)
        self._log_service.log_info('Generating plots')
        self._generate_common_plots()

    def _save_individual_experiments(self, result: Dict[ExperimentType, Dict[str, float]]):
        experiments_folder = self._file_service.get_experiments_path()

        for experiment_type, word_value_pairs in result.items():
            experiment_type_folder = self._file_service.combine_path(
                experiments_folder,
                experiment_type.value,
                create_if_missing=True)

            self._log_service.log_debug(
                f'Saving \'{experiment_type.value}\' experiment results at \'{experiment_type_folder}\'')

            values = [round(x, 1) for x in word_value_pairs.values()]
            if values is None or len(values) == 0:
                continue

            counter = Counter(values)
            filename = self._arguments_service.get_configuration_name()
            self._plot_service.plot_distribution(
                counts=counter,
                plot_options=PlotOptions(
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
            create_if_missing=True)

        ax = self._plot_service.create_plot()
        overlaps_by_config_and_seed = self._neighbourhood_overlap_process_service.get_calculated_overlaps()

        for configuration, overlaps_by_seed in overlaps_by_config_and_seed.items():
            combined_overlaps = self._neighbourhood_overlap_process_service.combine_seed_overlaps(overlaps_by_seed)
            value_summaries = self._neighbourhood_overlap_process_service.extract_value_summaries(combined_overlaps)

            for value_summary, overlap_line in value_summaries.items():
                ax = self._plot_service.plot_distribution(
                    counts=overlap_line,
                    plot_options=self._neighbourhood_overlap_process_service.get_distribution_plot_options(
                        ax,
                        configuration,
                        value_summary))

        self._plot_service.set_plot_properties(
            ax=ax,
            figure_options=FigureOptions(
                title=f'Neighbourhood overlaps ({self._arguments_service.language.value})'))

        self._plot_service.save_plot(
            save_path=experiment_type_folder,
            filename=f'combined-neighbourhood-overlaps-{self._arguments_service.language.value}')
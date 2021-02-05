from collections import defaultdict

from matplotlib.axes import Axes
from enums.value_summary import ValueSummary
from entities.plot.plot_options import PlotOptions
from entities.plot.figure_options import FigureOptions

import seaborn
from entities.cache.cache_options import CacheOptions
from enums.plot_legend_position import PlotLegendPosition
from entities.plot.legend_options import LegendOptions
from enums.configuration import Configuration
from entities.word_neighbourhood_stats import WordNeighbourhoodStats
import os
from services.tagging_service import TaggingService
from enums.part_of_speech import PartOfSpeech
from enums.ocr_output_type import OCROutputType
from services.vocabulary_service import VocabularyService
import numpy as np
from entities.word_evaluation import WordEvaluation
import math
from services.cache_service import CacheService
from overrides.overrides import overrides
from typing import Counter, List, Dict, Tuple
from overrides import overrides
from scipy.spatial import procrustes
from tqdm import tqdm

from enums.experiment_type import ExperimentType

from models.model_base import ModelBase

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.dataloader_service import DataLoaderService
from services.experiments.experiment_service_base import ExperimentServiceBase
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.plot_service import PlotService
from services.word_neighbourhood_service import WordNeighbourhoodService
from services.log_service import LogService

import pandas as pd

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
        #         callback_function=lambda: self._calculate_cosine_similarities(word_evaluations))
        #     self._log_service.log_info('Loaded cosine similarities')

        if ExperimentType.CosineDistance in experiment_types:
            result[ExperimentType.CosineDistance] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'cosine-distances{random_suffix}',
                    seed_specific=True),
                callback_function=lambda: self._calculate_cosine_distances(word_evaluations))

            self._log_service.log_info('Loaded cosine distances')

        if ExperimentType.NeighbourhoodOverlap in experiment_types:
            result[ExperimentType.NeighbourhoodOverlap] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'neighbourhood-overlaps{random_suffix}',
                    seed_specific=True),
                callback_function=lambda: self._generate_neighbourhood_similarity(word_evaluations))

            self._log_service.log_info('Loaded neighbourhood overlaps')

        if ExperimentType.CosineDistance in experiment_types and ExperimentType.NeighbourhoodOverlap in experiment_types:
            self._generate_neighbourhood_plots(
                word_evaluations,
                result[ExperimentType.CosineDistance])

        if ExperimentType.EuclideanDistance in experiment_types:
            result[ExperimentType.EuclideanDistance] = self._cache_service.get_item_from_cache(
                CacheOptions(
                    f'euclidean-distances{random_suffix}',
                    seed_specific=True),
                callback_function=lambda: self._calculate_euclidean_distances(word_evaluations))

            self._log_service.log_info('Loaded euclidean distances')

        # a, b, c = procrustes(model1_embeddings, model2_embeddings)

        self._log_service.log_info('Saving experiment results')
        self._save_experiment_results(result)

        self._log_service.log_info(
            'Experiments calculation completed successfully')

    def _generate_neighbourhood_plots(
            self,
            word_evaluations,
            cosine_distances: Dict[str, float]):
        target_tokens = self._get_target_tokens(cosine_distances)
        for target_token in target_tokens:
            i = next(i for i, word_evaluation in enumerate(
                word_evaluations) if word_evaluation.word == target_token)

            if i is None:
                continue

            word_evaluation = word_evaluations[i]
            remaining_words = word_evaluations[:i] + word_evaluations[i+1:]
            word_neighbourhood_stats = self._word_neighbourhood_service.get_word_neighbourhoods(
                word_evaluation,
                remaining_words,
                neighbourhood_set_size=50,
                include_embeddings=True)

            self._word_neighbourhood_service.plot_word_neighbourhoods(
                word_evaluation,
                word_neighbourhood_stats)

    def _calculate_cosine_similarities(self, word_evaluations: List[WordEvaluation]) -> Dict[str, float]:
        result = {}
        for word_evaluation in word_evaluations:
            if not word_evaluation.contains_all_embeddings():
                continue

            result[word_evaluation.word] = self._metrics_service.calculate_cosine_similarity(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

        return result

    def _calculate_cosine_distances(self, word_evaluations: List[WordEvaluation]) -> Dict[str, float]:
        result = {}
        for word_evaluation in word_evaluations:
            if not word_evaluation.contains_all_embeddings():
                continue

            cosine_distance = self._metrics_service.calculate_cosine_distance(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

            result[word_evaluation.word] = cosine_distance

        return result

    def _calculate_euclidean_distances(self, word_evaluations: List[WordEvaluation]) -> Dict[str, float]:
        result = {}

        for word_evaluation in word_evaluations:
            if not word_evaluation.contains_all_embeddings():
                continue

            result[word_evaluation.word] = self._metrics_service.calculate_euclidean_distance(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

        return result

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

        experiment_xlims = {
            ExperimentType.CosineDistance: (-0.05, 1.05),
            ExperimentType.NeighbourhoodOverlap: (-0.5, 10.5)
        }

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
            xlim = None
            if experiment_type in experiment_xlims.keys():
                xlim = experiment_xlims[experiment_type]

            filename = self._arguments_service.get_configuration_name()
            # self._plot_service.plot_distribution(
            #     counts=counter,
            #     title=experiment_type.value,
            #     save_path=experiment_type_folder,
            #     filename=filename,
            #     color='royalblue',
            #     fill=True,
            #     xlim=xlim)

    def _generate_common_plots(self):
        experiments_folder = self._file_service.get_experiments_path()
        experiment_type_folder = self._file_service.combine_path(
            experiments_folder,
            ExperimentType.NeighbourhoodOverlap.value,
            create_if_missing=True)

        ax = self._plot_service.create_plot()
        overlaps_by_config_and_seed = self._get_calculated_overlaps()

        for configuration, overlaps_by_seed in overlaps_by_config_and_seed.items():
            if all(x is None for x in overlaps_by_seed.values()):
                continue

            combined_overlaps = defaultdict(
                lambda: [None for _ in range(len(overlaps_by_seed.keys()))])

            for i, current_overlaps in enumerate(overlaps_by_seed.values()):
                if current_overlaps is None:
                    continue

                for overlap_amount in current_overlaps.values():
                    overlap_amount = int(overlap_amount / 10)

                    if combined_overlaps[overlap_amount][i] is None:
                        combined_overlaps[overlap_amount][i] = 0

                    combined_overlaps[overlap_amount][i] = combined_overlaps[overlap_amount][i] + 1

            value_summaries = {
                ValueSummary.Maximum: {},
                ValueSummary.Average: {},
                ValueSummary.Minimum: {},
            }
            for i in range(0, self._arguments_service.neighbourhood_set_size, 1):
                if i not in combined_overlaps.keys():
                    continue

                valid_overlaps = [x for x in combined_overlaps[i] if x is not None]

                value_summaries[ValueSummary.Minimum][i] = min(valid_overlaps)
                value_summaries[ValueSummary.Maximum][i] = max(valid_overlaps)
                value_summaries[ValueSummary.Average][i] = int(np.mean(valid_overlaps))

            for value_summary, overlap_line in value_summaries.items():
                ax = self._plot_service.plot_distribution(
                    counts=overlap_line,
                    plot_options=self._get_distribution_plot_options(
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

    def _get_calculated_overlaps(self) -> Dict[Configuration, Dict[int, dict]]:
        configurations = [
            Configuration.CBOW,
            Configuration.PPMI,
            Configuration.SkipGram,
            Configuration.BERT,
        ]

        seeds = [7, 13, 42]

        random_suffix = '-rnd' if self._arguments_service.initialize_randomly else ''
        cache_key = f'neighbourhood-overlaps{random_suffix}'

        result = {}

        for configuration in configurations:
            result[configuration] = {}

            for seed in seeds:
                config_overlaps = self._cache_service.get_item_from_cache(
                    CacheOptions(
                        cache_key,
                        configuration=configuration,
                        seed=seed))

                result[configuration][seed] = config_overlaps

        return result

    def _get_distribution_plot_options(
        self,
        ax: Axes,
        configuration: Configuration,
        value_summary: ValueSummary):
        alpha_values = {
            ValueSummary.Maximum: .3,
            ValueSummary.Average: 1,
            ValueSummary.Minimum: 1,
        }

        fill = {
            ValueSummary.Maximum: True,
            ValueSummary.Average: False,
            ValueSummary.Minimum: True,
        }

        linewidths = {
            ValueSummary.Maximum: 0,
            ValueSummary.Average: 1,
            ValueSummary.Minimum: 0,
        }

        colors = {
            Configuration.CBOW: {
                ValueSummary.Maximum: 'red',
                ValueSummary.Average: 'red',
                ValueSummary.Minimum: 'white',
            },
            Configuration.PPMI: {
                ValueSummary.Maximum: 'green',
                ValueSummary.Average: 'green',
                ValueSummary.Minimum: 'white',
            },
            Configuration.SkipGram: {
                ValueSummary.Maximum: 'darkblue',
                ValueSummary.Average: 'darkblue',
                ValueSummary.Minimum: 'white',
            },
            Configuration.BERT: {
                ValueSummary.Maximum: 'crimson',
                ValueSummary.Average: 'crimson',
                ValueSummary.Minimum: 'white',
            },
        }

        line_styles = {
            ValueSummary.Maximum: 'solid',
            ValueSummary.Average: 'dashed',
            ValueSummary.Minimum: 'solid',
        }

        result = PlotOptions(
            color=colors[configuration][value_summary],
            linestyle=line_styles[value_summary],
            fill=fill[value_summary],
            label=f'{configuration.value} [{value_summary.value}]',
            alpha=alpha_values[value_summary],
            line_width=linewidths[value_summary],
            ax=ax)

        return result

    def _generate_neighbourhood_similarity(
            self,
            word_evaluations: List[WordEvaluation]) -> Dict[str, int]:
        self._log_service.log_debug(
            'Generating neighbourhood similarity results')

        result = {}

        common_words_indices = [i for i, word_evaluation in enumerate(word_evaluations) if (
            word_evaluation.contains_all_embeddings() and word_evaluation.word not in result.keys())]

        for i in tqdm(iterable=common_words_indices, desc=f'Calculating neighbourhood overlaps', total=len(common_words_indices)):
            word_evaluation = word_evaluations[i]
            remaining_words = word_evaluations[:i] + word_evaluations[i+1:]
            word_neighbourhood_stats = self._word_neighbourhood_service.get_word_neighbourhoods(
                word_evaluation,
                remaining_words,
                neighbourhood_set_size=self._arguments_service.neighbourhood_set_size)

            result[word_evaluation.word] = word_neighbourhood_stats.overlaps_amount

            if i % 500 == 0:
                self._word_neighbourhood_service.cache_calculations()

        return result

    def _get_target_tokens(
            self,
            cosine_distances: Dict[str, float],
            pos_tags: List[PartOfSpeech] = [PartOfSpeech.Noun, PartOfSpeech.Verb]) -> List[str]:
        metric_results = [(word, distance)
                          for word, distance in cosine_distances.items()
                          if self._tagging_service.word_is_specific_tag(word, pos_tags)]

        metric_results.sort(key=lambda x: x[1], reverse=True)

        most_changed_100 = [result[0] for result in metric_results[-100:]]
        most_changed_100_string = ', '.join(most_changed_100)
        self._log_service.log_debug(
            f'Most changed 100 words: [{most_changed_100_string}]')

        most_changed = self._map_target_tokens(
            metric_results,
            targets_count=10)

        log_message = f'Target words to be used: [' + \
            ', '.join(most_changed) + ']'
        self._log_service.log_info(log_message)

        return most_changed

    def _map_target_tokens(
            self,
            ordered_tuples: List[Tuple[str, float]],
            targets_count: int) -> List[str]:
        result_tuples = []
        preferred_tokens = self._get_preferred_target_tokens()

        for tuple in ordered_tuples:
            if preferred_tokens is None or tuple[0] in preferred_tokens:
                result_tuples.append(tuple[0])

            if len(result_tuples) == targets_count:
                return result_tuples

        return result_tuples

    def _get_preferred_target_tokens(self) -> List[str]:
        preferred_tokens_path = os.path.join(
            self._file_service.get_experiments_path(),
            f'preferred-tokens-{self._arguments_service.language.value}.txt')

        if not os.path.exists(preferred_tokens_path):
            return None

        preferred_tokens = []
        with open(preferred_tokens_path, 'r', encoding='utf-8') as tokens_file:
            file_lines = tokens_file.readlines()
            if file_lines is None or len(file_lines) == 0:
                return None

            preferred_tokens = [x.strip().lower() for x in file_lines]

        return preferred_tokens

    def _get_word_evaluations_for_comparison(self, target_word: str, word_evaluations: List[List[WordEvaluation]]):
        if not self._arguments_service.separate_neighbourhood_vocabularies:
            remaining_words = [
                word_evaluation
                for word_evaluation in word_evaluations[0]
                if word_evaluation.word != target_word]

            return [remaining_words]

        words_1 = [
            word_evaluation
            for word_evaluation in word_evaluations[1]
            if word_evaluation.word != target_word]

        words_2 = [
            word_evaluation
            for word_evaluation in word_evaluations[2]
            if word_evaluation.word != target_word]

        result = [words_1, words_2]
        return result

import numpy as np
from entities.word_evaluation import WordEvaluation
import math
from services.cache_service import CacheService
from matplotlib.pyplot import xticks
from overrides.overrides import overrides
from typing import Counter, List, Dict, Tuple
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
from services.word_neighbourhood_service import WordNeighbourhoodService


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
            model: ModelBase):
        super().__init__(arguments_service, dataloader_service, file_service, model)

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._cache_service = cache_service
        self._word_neighbourhood_service = word_neighbourhood_service

    @overrides
    def execute_experiments(self, experiment_types: List[ExperimentType]):

        result = {experiment_type: {} for experiment_type in experiment_types}

        word_evaluations: List[WordEvaluation] = self._cache_service.get_item_from_cache(
            item_key='word-evaluations',
            callback_function=self._generate_embeddings)

        # if ExperimentType.CosineSimilarity in experiment_types:
        #     result[ExperimentType.CosineSimilarity] = self._cache_service.get_item_from_cache(
        #         item_key='cosine-similarities',
        #         callback_function=lambda: self._calculate_cosine_similarities(word_evaluations))

        if ExperimentType.CosineDistance in experiment_types:
            result[ExperimentType.CosineDistance] = self._cache_service.get_item_from_cache(
                item_key='cosine-distances',
                callback_function=lambda: self._calculate_cosine_distances(word_evaluations))

        if ExperimentType.EuclideanDistance in experiment_types:
            result[ExperimentType.EuclideanDistance] = self._cache_service.get_item_from_cache(
                item_key='euclidean-distances',
                callback_function=lambda: self._calculate_euclidean_distances(word_evaluations))

        # a, b, c = procrustes(model1_embeddings, model2_embeddings)

        self._generate_neighbourhood_similarity_results(
            result, word_evaluations)

        self._save_experiment_results(result)

    def _calculate_cosine_similarities(self, word_evaluations) -> Dict[str, float]:
        result = {}
        for word_evaluation in word_evaluations:
            result[word_evaluation.word] = self._metrics_service.calculate_cosine_similarity(
                list1=word_evaluation.embeddings_1,
                list2=word_evaluation.embeddings_2)

        return result

    def _calculate_cosine_distances(self, word_evaluations) -> Dict[str, float]:
        result = {}
        for word_evaluation in word_evaluations:
            result[word_evaluation.word] = self._metrics_service.calculate_cosine_distance(
                list1=word_evaluation.embeddings_1,
                list2=word_evaluation.embeddings_2)

        return result

    def _calculate_euclidean_distances(self, word_evaluations) -> Dict[str, float]:
        result = {}

        for word_evaluation in word_evaluations:
            result[word_evaluation.word] = self._metrics_service.calculate_euclidean_distance(
                list1=word_evaluation.embeddings_1,
                list2=word_evaluation.embeddings_2)

        return result

    def _generate_embeddings(self) -> List[WordEvaluation]:
        result = []

        dataloader_length = len(self._dataloader)
        for i, batch in enumerate(self._dataloader):
            print(f'{i}/{dataloader_length}         \r', end='')

            words, token_ids = batch
            outputs = self._model.get_embeddings(token_ids)

            result.extend([WordEvaluation(word, embeddings_1, embeddings_2)
                           for word, embeddings_1, embeddings_2 in zip(words, outputs[0], outputs[1])])

        return result

    def _save_experiment_results(self, result: Dict[ExperimentType, Dict[str, float]]):
        experiments_folder = self._file_service.get_experiments_path()
        distances_folder = self._file_service.combine_path(
            experiments_folder, 'distances', self._arguments_service.language.value, create_if_missing=True)

        for experiment_type, word_value_pairs in result.items():
            values = [round(x, 1) for x in word_value_pairs.values()]

            if values is None or len(values) == 0:
                continue

            counter = Counter(values)

            filename = f'{self._arguments_service.configuration.value}-{experiment_type.value}'
            self._plot_service.plot_counters_histogram(
                counter_labels=['a'],
                counters=[counter],
                title=experiment_type.value,
                show_legend=False,
                counter_colors=['royalblue'],
                bars_padding=0,
                plot_values_above_bars=True,
                save_path=distances_folder,
                filename=filename)

    def _generate_neighbourhood_similarity_results(self, result: Dict[ExperimentType, Dict[str, float]], word_evaluations: List[WordEvaluation]):
        most_changed = self._get_most_changed_words(result)
        for (changed_word, _) in most_changed:
            target_word = next(
                (w for w in word_evaluations if w.word == changed_word), None)
            if target_word is None:
                raise Exception('Could not find target word')

            remaining_words = [
                word_evaluation
                for word_evaluation in word_evaluations
                if word_evaluation.word != target_word.word]

            word_neighbourhoods = self._word_neighbourhood_service.get_word_neighbourhoods(
                target_word, remaining_words)

            self._word_neighbourhood_service.plot_word_neighbourhoods(
                target_word,
                word_neighbourhoods=list(word_neighbourhoods))

    def _get_most_changed_words(self, result, metric: ExperimentType = ExperimentType.CosineDistance) -> List[Tuple[str, float]]:
        if metric not in result.keys():
            raise Exception(f'Metric {metric} not calculated')

        metric_results = [(word, distance)
                          for word, distance in result[metric].items()]
        metric_results.sort(key=lambda x: x[1])

        most_changed = metric_results[-15:][::-1]
        return most_changed

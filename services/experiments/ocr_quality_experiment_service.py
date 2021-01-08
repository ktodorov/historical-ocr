from enums.ocr_output_type import OCROutputType
from services.vocabulary_service import VocabularyService
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
            vocabulary_service: VocabularyService,
            model: ModelBase):
        super().__init__(arguments_service, dataloader_service, file_service, model)

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._cache_service = cache_service
        self._word_neighbourhood_service = word_neighbourhood_service
        self._vocabulary_service = vocabulary_service

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

            result[word_evaluation.word] = self._metrics_service.calculate_cosine_distance(
                list1=word_evaluation.get_embeddings(idx=0),
                list2=word_evaluation.get_embeddings(idx=1))

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

        dataloader_length = len(self._dataloader)
        for i, batch in enumerate(self._dataloader):
            print(f'{i}/{dataloader_length}         \r', end='')

            tokens, vocab_ids = batch
            outputs = self._model.get_embeddings(tokens, vocab_ids)
            result.extend(outputs)

        if self._arguments_service.separate_neighbourhood_vocabularies:
            processed_tokens = [we.word for we in result]
            for ocr_output_type in [OCROutputType.Raw, OCROutputType.GroundTruth]:
                vocab_key = f'vocab-{ocr_output_type.value}'
                self._vocabulary_service.load_cached_vocabulary(vocab_key)
                unprocessed_tokens = []
                vocabulary_size = self._vocabulary_service.vocabulary_size()
                for i, (_, token) in enumerate(self._vocabulary_service.get_vocabulary_tokens(exclude_special_tokens=True)):
                    print(f'{ocr_output_type.value}: {i}/{vocabulary_size}         \r', end='')
                    if token in processed_tokens:
                        continue

                    unprocessed_tokens.append(token)

                batch_size = self._arguments_service.batch_size
                for i in range(0, len(unprocessed_tokens), batch_size):
                    tokens = unprocessed_tokens[i:i+batch_size]
                    word_evaluations = self._model.get_embeddings(tokens, vocab_ids=None, skip_unknown=True)
                    result.extend(word_evaluations)
                    processed_tokens.extend(tokens)

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
                word_neighbourhoods=word_neighbourhoods)

    def _get_most_changed_words(self, result, metric: ExperimentType = ExperimentType.CosineDistance) -> List[Tuple[str, float]]:
        if metric not in result.keys():
            raise Exception(f'Metric {metric} not calculated')

        metric_results = [(word, distance)
                          for word, distance in result[metric].items()]
        metric_results.sort(key=lambda x: x[1])

        most_changed = metric_results[-10:][::-1]
        return most_changed

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

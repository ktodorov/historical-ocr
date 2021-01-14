import os
from services.tagging_service import TaggingService
from enums.part_of_speech import PartOfSpeech
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
from tqdm import tqdm
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
from services.log_service import LogService

from utils.dict_utils import stringify_dictionary


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

        word_evaluations: List[WordEvaluation] = self._cache_service.get_item_from_cache(
            item_key='word-evaluations',
            callback_function=self._generate_embeddings)

        self._log_service.log_debug('Loaded word evaluations')

        # if ExperimentType.CosineSimilarity in experiment_types:
        #     result[ExperimentType.CosineSimilarity] = self._cache_service.get_item_from_cache(
        #         item_key='cosine-similarities',
        #         callback_function=lambda: self._calculate_cosine_similarities(word_evaluations))
        #     self._log_service.log_debug('Loaded cosine similarities')

        if ExperimentType.CosineDistance in experiment_types:
            result[ExperimentType.CosineDistance] = self._cache_service.get_item_from_cache(
                item_key='cosine-distances',
                callback_function=lambda: self._calculate_cosine_distances(word_evaluations))

            self._log_service.log_debug('Loaded cosine distances')

        if ExperimentType.EuclideanDistance in experiment_types:
            result[ExperimentType.EuclideanDistance] = self._cache_service.get_item_from_cache(
                item_key='euclidean-distances',
                callback_function=lambda: self._calculate_euclidean_distances(word_evaluations))

            self._log_service.log_debug('Loaded euclidean distances')

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
        experiments_folder = self._file_service.get_experiments_path()
        distances_folder = self._file_service.combine_path(
            experiments_folder, 'distances', create_if_missing=True)

        self._log_service.log_debug(
            f'Saving experiment results [Distances] at \'{distances_folder}\'')

        for experiment_type, word_value_pairs in result.items():
            values = [round(x, 1) for x in word_value_pairs.values()]

            if values is None or len(values) == 0:
                continue

            counter = Counter(values)

            filename = f'{self._arguments_service.get_configuration_name()}-{experiment_type.value}'
            self._log_service.log_debug(
                f'Saving {experiment_type} as {filename}')
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
        self._log_service.log_debug(
            'Generating neighbourhood similarity results')

        target_tokens = self._get_target_tokens(result)
        for (target_token, _) in tqdm(iterable=target_tokens, desc='Generating neighbourhood plots', total=len(target_tokens)):
            target_word_evaluation = next(
                (w for w in word_evaluations if w.word == target_token), None)
            if target_word_evaluation is None:
                raise Exception('Could not find target word')

            remaining_words = [
                word_evaluation
                for word_evaluation in word_evaluations
                if word_evaluation.word != target_word_evaluation.word]

            word_neighbourhoods = self._word_neighbourhood_service.get_word_neighbourhoods(
                target_word_evaluation, remaining_words)

            self._word_neighbourhood_service.plot_word_neighbourhoods(
                target_word_evaluation,
                word_neighbourhoods=word_neighbourhoods)

    def _get_target_tokens(
            self,
            result,
            metric: ExperimentType = ExperimentType.CosineDistance,
            pos_tags: List[PartOfSpeech] = [PartOfSpeech.Noun, PartOfSpeech.Verb]) -> List[Tuple[str, float]]:
        if metric not in result.keys():
            raise Exception(f'Metric {metric} not calculated')

        metric_results = [(word, distance)
                          for word, distance in result[metric].items()
                          if self._tagging_service.word_is_specific_tag(word, pos_tags)]

        metric_results.sort(key=lambda x: x[1])

        top_100_most_changed_words = [result[0] for result in metric_results[-100:][::-1]]
        top_100_string = ', '.join(top_100_most_changed_words)
        self._log_service.log_debug(f'Top 100 most changed words: [{top_100_string}]')

        most_changed = self._map_target_tokens(metric_results[::-1], targets_count=10)

        log_message = f'Target words to be used: [' + \
            ', '.join([x[0] for x in most_changed]) + ']'
        self._log_service.log_info(log_message)

        return most_changed

    def _map_target_tokens(
        self,
        ordered_tuples: List[Tuple[str, float]],
        targets_count: int):
        result_tuples = []
        preferred_tokens = self._get_preferred_target_tokens()

        for tuple in ordered_tuples:
            if preferred_tokens is None or tuple[0] in preferred_tokens:
                result_tuples.append(tuple)

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

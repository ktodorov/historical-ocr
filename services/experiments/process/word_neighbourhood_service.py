from services.experiments.process.neighbourhood_similarity_process_service import NeighbourhoodSimilarityProcessService
from enums.font_weight import FontWeight
from entities.plot.label_options import LabelOptions
from entities.plot.figure_options import FigureOptions
from entities.plot.plot_options import PlotOptions
from entities.cache.cache_options import CacheOptions
from services.cache_service import CacheService
from scipy import sparse
from scipy.sparse import vstack
from tqdm import tqdm

from entities.word_neighbourhood_stats import WordNeighbourhoodStats
from services.log_service import LogService
from entities.plot.legend_options import LegendOptions
from typing import Dict, List, Tuple
from matplotlib.pyplot import plot
import math
import numpy as np

from entities.word_evaluation import WordEvaluation

from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.plot_service import PlotService
from services.fit_transformation_service import FitTransformationService


class WordNeighbourhoodService:
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            metrics_service: MetricsService,
            plot_service: PlotService,
            file_service: FileService,
            log_service: LogService,
            fit_transformation_service: FitTransformationService,
            cache_service: CacheService,
            neighbourhood_similarity_process_service: NeighbourhoodSimilarityProcessService):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._file_service = file_service
        self._log_service = log_service
        self._fit_transformation_service = fit_transformation_service
        self._cache_service = cache_service
        self._neighbourhood_similarity_process_service = neighbourhood_similarity_process_service

        self._word_similarities_cache_options = CacheOptions(
            'word-similarities',
            seed_specific=True,
            key_suffixes=[
                '-sep' if self._arguments_service.separate_neighbourhood_vocabularies else '',
                '-min',
                str(self._arguments_service.minimal_occurrence_limit)
            ])

        self._word_similarity_indices: Dict[str, Dict[int, list]] = self._cache_service.get_item_from_cache(
            self._word_similarities_cache_options,
            callback_function=lambda: {})

    def plot_word_neighbourhoods(
            self,
            target_word_evaluation: WordEvaluation,
            word_neighbourhood_stats: WordNeighbourhoodStats):

        self._log_service.log_debug(
            f'Plotting neighbourhoods for word \'{target_word_evaluation.word}\'')

        all_words = word_neighbourhood_stats.get_all_words()

        all_word_embeddings = []
        for i in range(word_neighbourhood_stats.neighbourhoods_amount):
            all_word_embeddings.append(
                target_word_evaluation.get_embeddings(i))

        all_word_embeddings.extend(
            word_neighbourhood_stats.get_all_embeddings())

        assert all(not np.isnan(x).any()
                   for x in all_word_embeddings), "Invalid values found in word embeddings"

        fitted_result = self._fit_transformation_service.fit_and_transform_vectors(
            number_of_components=word_neighbourhood_stats.neighbourhoods_amount,
            vectors=all_word_embeddings)

        self._plot_fitted_result(
            fitted_result[:word_neighbourhood_stats.neighbourhoods_amount],
            fitted_result[word_neighbourhood_stats.neighbourhoods_amount:],
            target_word_evaluation,
            all_words,
            word_neighbourhood_stats)

    def get_word_neighbourhoods(
            self,
            word_evaluation: WordEvaluation,
            vocabulary_evaluations: List[WordEvaluation],
            neighbourhood_set_size: int,
            models_count: int = 2,
            include_embeddings: bool = False) -> WordNeighbourhoodStats:
        self._log_service.log_debug(
            f'Extracting neighbourhoods for word \'{word_evaluation.word}\'')
        result = WordNeighbourhoodStats(
            word_evaluation.word, neighbourhoods=[])
        for i in range(models_count):
            model_evaluations = [
                vocabulary_evaluation
                for vocabulary_evaluation in vocabulary_evaluations
                if vocabulary_evaluation.token_is_known(idx=i)]

            word_neighbourhood = self._get_word_neighbourhood(
                word_evaluation,
                model_evaluations,
                embeddings_idx=i,
                neighbourhood_set_size=neighbourhood_set_size,
                output_full_evaluations=include_embeddings)

            result.add_neighbourhood(word_neighbourhood)

        return result

    def _plot_fitted_result(
            self,
            target_word_fitted_vectors: np.ndarray,
            fitted_vectors: np.ndarray,
            target_word_evaluation: WordEvaluation,
            all_words: List[str],
            word_neighbourhoods: WordNeighbourhoodStats):
        ax = self._plot_service.create_plot()

        labels_colors = ['crimson', 'royalblue', 'darkgreen']
        word_neighbourhood_length = word_neighbourhoods.neighbourhood_size

        plot_options = PlotOptions(
            ax=ax,
            figure_options=FigureOptions(
                show_plot=False))

        for i in range(word_neighbourhoods.neighbourhoods_amount):
            target_word_fitted_vector = target_word_fitted_vectors[i]
            current_fitted_vectors = fitted_vectors[(
                i * word_neighbourhood_length):(i+1)*word_neighbourhood_length]

            x_coords = target_word_fitted_vector[0] + \
                current_fitted_vectors[:, 0]
            y_coords = target_word_fitted_vector[1] + \
                current_fitted_vectors[:, 1]

            self._plot_service.plot_scatter(
                x_coords,
                y_coords,
                plot_options=plot_options)

            current_words = [target_word_evaluation.word] + all_words[(
                i*word_neighbourhood_length):((i+1)*word_neighbourhood_length)]
            current_word_colors = [labels_colors[i]] + [labels_colors[i] if all_words.count(
                x) == 1 else labels_colors[-1] for x in current_words[1:]]

            labels_options = [
                LabelOptions(
                    x=x_coords[i],
                    y=y_coords[i],
                    text=current_words[i],
                    text_color=current_word_colors[i])
                for i in range(word_neighbourhood_length + 1)]

            labels_options[0].font_weight = FontWeight.Bold
            labels_options[0].font_size = 15

            self._plot_service.plot_labels(labels_options, plot_options)

        self._plot_service.set_plot_properties(
            ax=ax,
            figure_options=FigureOptions(
                title=f'Neighbourhoods `{target_word_evaluation.word}`',
                hide_axis=True),
            legend_options=LegendOptions(
                show_legend=True,
                legend_colors=labels_colors,
                legend_labels=['raw', 'ground truth', 'overlapping']))

        experiments_folder = self._file_service.get_experiments_path()
        neighbourhoods_folder = self._file_service.combine_path(
            experiments_folder,
            'neighbourhoods',
            self._arguments_service.get_configuration_name(),
            create_if_missing=True)

        self._plot_service.save_plot(
            save_path=neighbourhoods_folder,
            filename=f'{target_word_evaluation}-neighborhood-change')

    def _get_word_neighbourhood(
            self,
            word_evaluation: WordEvaluation,
            model_evaluations: List[WordEvaluation],
            embeddings_idx: int,
            neighbourhood_set_size: int,
            output_full_evaluations: bool = False) -> List[WordEvaluation]:
        if (not output_full_evaluations and word_evaluation.word in self._word_similarity_indices.keys() and embeddings_idx in self._word_similarity_indices[word_evaluation.word]):
            indices = self._word_similarity_indices[word_evaluation.word][embeddings_idx]
        else:
            target_embeddings = np.array(
                [word_evaluation.get_embeddings(embeddings_idx)])
            model_embeddings = np.array([model_evaluation.get_embeddings(
                embeddings_idx) for model_evaluation in model_evaluations])

            distances = self._metrics_service.calculate_cosine_similarities(
                target_embeddings, model_embeddings)

            indices = np.argsort(distances.squeeze())

            if not output_full_evaluations:
                if word_evaluation.word not in self._word_similarity_indices.keys():
                    self._word_similarity_indices[word_evaluation.word] = {}

                self._word_similarity_indices[word_evaluation.word][embeddings_idx] = indices

        if neighbourhood_set_size > len(indices):
            self._log_service.log_warning(
                f'Neighbourhood set size ({neighbourhood_set_size}) is larger than the collection ({len(indices)}). Using the entire collection instead')
            neighbourhood_set_size = len(indices)

        max_indices = indices[:neighbourhood_set_size]

        if output_full_evaluations:
            result_evaluations = [x for i, x in enumerate(
                model_evaluations) if i in max_indices]
            return result_evaluations

        return max_indices

    def generate_neighbourhood_plots(
            self,
            word_evaluations,
            cosine_distances: Dict[str, float]):
        target_tokens = self._neighbourhood_similarity_process_service.get_target_tokens(cosine_distances)
        for target_token in target_tokens:
            i = next(i for i, word_evaluation in enumerate(
                word_evaluations) if word_evaluation.word == target_token)

            if i is None:
                continue

            word_evaluation = word_evaluations[i]
            remaining_words = word_evaluations[:i] + word_evaluations[i+1:]
            word_neighbourhood_stats = self.get_word_neighbourhoods(
                word_evaluation,
                remaining_words,
                neighbourhood_set_size=50,
                include_embeddings=True)

            self.plot_word_neighbourhoods(
                word_evaluation,
                word_neighbourhood_stats)

    def generate_neighbourhood_similarities(
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
            word_neighbourhood_stats = self.get_word_neighbourhoods(
                word_evaluation,
                remaining_words,
                neighbourhood_set_size=self._arguments_service.neighbourhood_set_size)

            result[word_evaluation.word] = word_neighbourhood_stats.overlaps_amount

            if i % 500 == 0:
                self._cache_calculations()

        return result

    def _cache_calculations(self):
        self._cache_service.cache_item(
            self._word_similarity_indices,
            self._word_similarities_cache_options)

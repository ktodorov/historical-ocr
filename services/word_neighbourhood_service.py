from scipy import sparse
from scipy.sparse import vstack

from entities.word_neighbourhood_stats import WordNeighbourhoodStats
from services.log_service import LogService
from entities.plot.legend_options import LegendOptions
from typing import List, Tuple
from matplotlib.pyplot import plot
import math
import numpy as np

from entities.word_evaluation import WordEvaluation

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.plot_service import PlotService
from services.fit_transformation_service import FitTransformationService


class WordNeighbourhoodService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            metrics_service: MetricsService,
            plot_service: PlotService,
            file_service: FileService,
            log_service: LogService,
            fit_transformation_service: FitTransformationService):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._file_service = file_service
        self._log_service = log_service
        self._fit_transformation_service = fit_transformation_service

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
            models_count: int = 2) -> WordNeighbourhoodStats:
        self._log_service.log_debug(
            f'Extracting neighbourhoods for word \'{word_evaluation.word}\'')
        result = WordNeighbourhoodStats(word_evaluation.word, neighbourhoods=[])
        for i in range(models_count):
            model_evaluations = [
                vocabulary_evaluation
                for vocabulary_evaluation in vocabulary_evaluations
                if vocabulary_evaluation.token_is_known(idx=i)]

            word_neighbourhood = self._get_word_neighbourhood(
                word_evaluation,
                model_evaluations,
                embeddings_idx=i)

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
        bold_mask = [False for _ in range(word_neighbourhood_length + 1)]
        bold_mask[0] = True
        font_sizes = [None for _ in range(word_neighbourhood_length + 1)]
        font_sizes[0] = 15

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
                color='white',
                ax=ax,
                show_plot=False)

            current_words = [target_word_evaluation.word] + all_words[(
                i*word_neighbourhood_length):((i+1)*word_neighbourhood_length)]
            current_word_colors = [labels_colors[i]] + [labels_colors[i] if all_words.count(
                x) == 1 else labels_colors[-1] for x in current_words[1:]]

            self._plot_service.plot_labels(
                x_coords,
                y_coords,
                current_words,
                colors=current_word_colors,
                ax=ax,
                show_plot=False,
                bold_mask=bold_mask,
                font_sizes=font_sizes)

        self._plot_service.set_plot_properties(
            ax=ax,
            title=f'Neighbourhoods `{target_word_evaluation.word}`',
            hide_axis=True,
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
            embeddings_idx: int) -> List[WordEvaluation]:
        target_embeddings = np.array([word_evaluation.get_embeddings(embeddings_idx)])
        model_embeddings = np.array([model_evaluation.get_embeddings(embeddings_idx) for model_evaluation in model_evaluations])

        distances = self._metrics_service.calculate_cosine_similarities(target_embeddings, model_embeddings)

        # distances = [
        #     self._metrics_service.calculate_cosine_distance(
        #         word_evaluation.get_embeddings(embeddings_idx),
        #         model_evaluation.get_embeddings(embeddings_idx))
        #     for model_evaluation in model_evaluations
        # ]

        indices = np.argsort(distances.squeeze())

        max_indices = indices[:10]
        result_words = [model_evaluations[i] for i in max_indices]
        return result_words

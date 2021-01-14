from services.log_service import LogService
from entities.plot.legend_options import LegendOptions
from typing import List, Tuple
from matplotlib.pyplot import plot
import math
import numpy as np

from MulticoreTSNE import MulticoreTSNE as TSNE

from entities.word_evaluation import WordEvaluation

from services.arguments.arguments_service_base import ArgumentsServiceBase
from services.file_service import FileService
from services.metrics_service import MetricsService
from services.plot_service import PlotService


class WordNeighbourhoodService:
    def __init__(
            self,
            arguments_service: ArgumentsServiceBase,
            metrics_service: MetricsService,
            plot_service: PlotService,
            file_service: FileService,
            log_service: LogService):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._file_service = file_service
        self._log_service = log_service

    def plot_word_neighbourhoods(
            self,
            target_word: WordEvaluation,
            word_neighbourhoods: List[List[WordEvaluation]]):

        self._log_service.log_debug(f'Plotting neighbourhoods for word \'{target_word.word}\'')

        all_words = []
        all_word_embeddings = []
        for i, word_neighbourhood in enumerate(word_neighbourhoods):
            all_words.append(target_word.word)
            all_words.extend(
                [w.word for w in word_neighbourhood])
            all_word_embeddings.append(target_word.get_embeddings(i))
            all_word_embeddings.extend(
                [w.get_embeddings(i) for w in word_neighbourhood])

        assert all(not np.isnan(x).any()
                   for x in all_word_embeddings), "Invalid values found in word embeddings"

        tsne = TSNE(n_components=2, random_state=0, n_jobs=4)
        tsne_result = tsne.fit_transform(np.array(all_word_embeddings))
        self._plot_tsne_result(
            tsne_result,
            target_word,
            all_words,
            word_neighbourhoods)

    def get_word_neighbourhoods(
            self,
            target_word: WordEvaluation,
            all_words: List[WordEvaluation],
            models_count: int = 2) -> Tuple[List[WordEvaluation], List[WordEvaluation]]:

        self._log_service.log_debug(f'Extracting neighbourhoods for word \'{target_word.word}\'')
        result = []
        for i in range(models_count):
            neighbourhood = self._get_word_neighbourhood(
                target_word,
                all_words,
                embeddings_idx=i)

            result.append(neighbourhood)

        return result

    def _plot_tsne_result(
            self,
            tsne_result,
            target_word: WordEvaluation,
            all_words: List[str],
            word_neighbourhoods: List[List[WordEvaluation]]):
        ax = self._plot_service.create_plot()

        labels_colors = ['crimson', 'royalblue', 'darkgreen']
        word_neighbourhood_length = len(word_neighbourhoods[0]) + 1
        for i in range(len(word_neighbourhoods)):
            x_coords = tsne_result[(
                i*word_neighbourhood_length):((i+1)*word_neighbourhood_length), 0]
            y_coords = tsne_result[(
                i*word_neighbourhood_length):((i+1)*word_neighbourhood_length), 1]
            bold_mask = [False for _ in range(len(x_coords))]
            bold_mask[0] = True

            font_sizes = [None for _ in range(len(x_coords))]
            font_sizes[0] = 15

            self._plot_service.plot_scatter(
                x_coords,
                y_coords,
                color='white',
                ax=ax,
                show_plot=False)

            current_words = all_words[(i*word_neighbourhood_length):((i+1)*word_neighbourhood_length)]
            current_word_colors = [labels_colors[i]] + [labels_colors[i] if all_words.count(x) == 1 else labels_colors[-1] for x in current_words[1:]]
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
            title=f'Neighbourhoods `{target_word.word}`',
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
            filename=f'{target_word}-neighborhood-change')

    def _get_word_neighbourhood(
            self,
            target_word: WordEvaluation,
            all_words: List[WordEvaluation],
            embeddings_idx: int) -> List[WordEvaluation]:
        distances = [
            self._metrics_service.calculate_cosine_distance(
                target_word.get_embeddings(embeddings_idx),
                word_evaluation.get_embeddings(embeddings_idx))
            if word_evaluation.token_is_known(idx=embeddings_idx) else (math.inf)
            for word_evaluation in all_words
        ]

        indices = np.argsort(distances)

        max_indices = indices[:10]
        result_words = [all_words[i] for i in max_indices]
        return result_words

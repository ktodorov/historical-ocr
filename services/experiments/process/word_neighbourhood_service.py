from enums.configuration import Configuration
from enums.word_evaluation_type import WordEvaluationType
from enums.overlap_type import OverlapType
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
from services.process.evaluation_process_service import EvaluationProcessService


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
            neighbourhood_similarity_process_service: NeighbourhoodSimilarityProcessService,
            process_service: EvaluationProcessService):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._plot_service = plot_service
        self._file_service = file_service
        self._log_service = log_service
        self._fit_transformation_service = fit_transformation_service
        self._cache_service = cache_service
        self._neighbourhood_similarity_process_service = neighbourhood_similarity_process_service

        # load previously cached word similarity calculations
        common_tokens = process_service.get_common_words()
        self._word_similarity_indices, self._cache_needs = self._load_cached_calculations(
            common_tokens)

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
            neighbourhood_set_sizes: List[int],
            overlap_type: OverlapType,
            include_embeddings: bool = False) -> Dict[int, WordNeighbourhoodStats]:
        self._log_service.log_debug(
            f'Extracting neighbourhoods for word \'{word_evaluation.word}\'')
        result = {
            neighbourhood_set_size: WordNeighbourhoodStats(
                word_evaluation.word, neighbourhoods=[])
            for neighbourhood_set_size in neighbourhood_set_sizes
        }

        model_indices = []
        if overlap_type == OverlapType.BASEvsGT:
            model_indices = [2, 1]
        elif overlap_type == OverlapType.BASEvsOCR:
            model_indices = [2, 0]
        elif overlap_type == OverlapType.BASEvsOG:
            model_indices = [2, 3]
        elif overlap_type == OverlapType.GTvsOCR:
            model_indices = [1, 0]

        for i in model_indices:
            word_neighbourhoods_per_set_size = self._get_word_neighbourhood(
                word_evaluation,
                vocabulary_evaluations,
                embeddings_idx=i,
                neighbourhood_set_sizes=neighbourhood_set_sizes,
                output_full_evaluations=include_embeddings)

            for neighbourhood_set_size, word_neighbourhood in word_neighbourhoods_per_set_size.items():
                result[neighbourhood_set_size].add_neighbourhood(word_neighbourhood)

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
            legend_options=LegendOptions(show_legend=False),
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
                    x=x_coords[k],
                    y=y_coords[k],
                    text=current_words[k],
                    text_color=current_word_colors[k])
                for k in range(word_neighbourhood_length)]

            labels_options[0]._font_weight = FontWeight.Bold
            labels_options[0]._font_size = 15

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

    def _token_already_calculated(self, token: str, embeddings_idx: int) -> bool:
        if (token not in self._word_similarity_indices.keys() or
            embeddings_idx not in self._word_similarity_indices[token].keys() or
                self._word_similarity_indices[token][embeddings_idx] is None):
            return False

        return True

    def _get_word_neighbourhood(
            self,
            word_evaluation: WordEvaluation,
            model_evaluations: List[WordEvaluation],
            embeddings_idx: int,
            neighbourhood_set_sizes: List[int],
            output_full_evaluations: bool = False) -> Dict[int, List[WordEvaluation]]:
        # We check if we have already calculated this word neighbourhood for the selected embeddings id
        if (not output_full_evaluations and self._token_already_calculated(word_evaluation.word, embeddings_idx)):
            indices = self._word_similarity_indices[word_evaluation.word][embeddings_idx]
        else:
            # If no calculation is available, we calculate and cache
            target_embeddings = np.array(
                [word_evaluation.get_embeddings(embeddings_idx)])
            model_embeddings = np.array([model_evaluation.get_embeddings(
                embeddings_idx) for model_evaluation in model_evaluations])

            distances = self._metrics_service.calculate_cosine_similarities(
                target_embeddings, model_embeddings)

            indices = np.argsort(distances.squeeze())[::-1]

            if not output_full_evaluations:
                self._cache_needs[WordEvaluationType(embeddings_idx)] = True
                if word_evaluation.word not in self._word_similarity_indices.keys():
                    self._word_similarity_indices[word_evaluation.word] = {}

                # We mark the indices to be cached because we add a new entry
                self._word_similarity_indices[word_evaluation.word][embeddings_idx] = indices

        result = {}
        for neighbourhood_set_size in neighbourhood_set_sizes:
            if neighbourhood_set_size > len(indices):
                self._log_service.log_error(
                    f'Neighbourhood set size ({neighbourhood_set_size}) is larger than the collection ({len(indices)}). Using the entire collection instead')
                raise Exception('Invalid set size')

            max_indices = indices[:neighbourhood_set_size]

            if output_full_evaluations:
                result_evaluations = [x for i, x in enumerate(
                    model_evaluations) if i in max_indices]
                result[neighbourhood_set_size] = result_evaluations
                continue

            result[neighbourhood_set_size] = max_indices

        return result

    def generate_neighbourhood_plots(
            self,
            word_evaluations,
            cosine_distances: Dict[str, float]):
        target_tokens = self._neighbourhood_similarity_process_service.get_target_tokens(
            cosine_distances)
        neighbourhood_set_size = 50

        for target_token in tqdm(target_tokens, desc="Processing target tokens", total=len(target_tokens)):
            i = next(i for i, word_evaluation in enumerate(
                word_evaluations) if word_evaluation.word == target_token)

            if i is None:
                continue

            word_evaluation = word_evaluations[i]
            remaining_words = [word_evaluation for idx, word_evaluation in enumerate(
                word_evaluations) if word_evaluation.contains_all_embeddings(OverlapType.GTvsOCR) and idx != i]
            word_neighbourhood_stats = self.get_word_neighbourhoods(
                word_evaluation,
                remaining_words,
                neighbourhood_set_sizes=[neighbourhood_set_size],
                overlap_type=OverlapType.GTvsOCR,
                include_embeddings=True)

            self.plot_word_neighbourhoods(
                word_evaluation,
                word_neighbourhood_stats[neighbourhood_set_size])

    def generate_neighbourhood_similarities(
            self,
            word_evaluations: List[WordEvaluation],
            overlap_type: OverlapType) -> Dict[str, int]:
        self._log_service.log_debug(
            f'Generating neighbourhood similarity results for overlap type \'{overlap_type.value}\'')

        # get all indices of words that support the current overlap type
        common_words_indices = [
            i
            for i, word_evaluation in enumerate(word_evaluations)
            if word_evaluation.contains_all_embeddings(overlap_type)]

        percentages = list(range(1, 101, 1))  # 1..20
        words_amounts = [
            int(len(common_words_indices) * (float(percentage)/ 100)) - (1 if percentage == 100 else 0)
            for percentage in percentages]

        result = {
            percentage: {}
            for percentage in percentages
        }

        self._log_service.log_summary(
            f'Total \'{overlap_type.value}\' neighbourhood overlaps', len(common_words_indices))
        for i in tqdm(iterable=common_words_indices, desc=f'Calculating neighbourhood overlaps [\'{overlap_type.value}\']', total=len(common_words_indices)):
            # get the target word evaluation
            word_evaluation = word_evaluations[i]

            # get the remaining valid word evaluations
            remaining_words = [word_evaluations[idx]
                               for idx in common_words_indices if idx != i]

            # calculate the word neighbourhood stats for this word
            word_neighbourhood_stats_per_set_size = self.get_word_neighbourhoods(
                word_evaluation,
                remaining_words,
                neighbourhood_set_sizes=words_amounts,
                overlap_type=overlap_type)

            # we only need the overlaps amount
            for words_amount, percentage in zip(words_amounts, percentages):
                result[percentage][word_evaluation.word] = word_neighbourhood_stats_per_set_size[words_amount].overlaps_amount

            # occasionally cache the calculations performed so far in case the process is interrupted
            if i % 500 == 0:
                self._log_service.log_summary(
                    f'Processed \'{overlap_type.value}\' neighbourhood overlaps', i)
                self._save_calculations()

        self._save_calculations()

        return result

    def _load_cached_calculations(self, common_tokens: List[str]) -> Dict[str, Dict[int, list]]:
        result = {token: {} for token in common_tokens}
        cache_needs = {}

        for i, word_evaluation_type in enumerate(WordEvaluationType):
            cache_needs[word_evaluation_type] = False
            word_similarities_cache_options = self._create_cache_options(
                word_evaluation_type)

            current_word_similarity_indices: Dict[str, Dict[int, list]] = self._cache_service.get_item_from_cache(
                word_similarities_cache_options)

            if current_word_similarity_indices is None:
                cache_needs[word_evaluation_type] = True
                continue

            for token, value in current_word_similarity_indices.items():
                if token not in result.keys():
                    result[token] = {}

                result[token][i] = value

        return result, cache_needs

    def _save_calculations(self):
        for i, word_evaluation_type in enumerate(WordEvaluationType):
            if not self._cache_needs[word_evaluation_type]:
                continue

            cache_options = self._create_cache_options(word_evaluation_type)
            current_value = {token: embeddings[i] if i in embeddings.keys(
            ) else None for token, embeddings in self._word_similarity_indices.items()}

            self._cache_service.cache_item(
                current_value,
                cache_options)

            self._cache_needs[word_evaluation_type] = False

    def _create_cache_options(self, word_evaluation_type: WordEvaluationType):
        random_suffix = ''
        if word_evaluation_type == WordEvaluationType.Baseline or self._arguments_service.initialize_randomly:
            random_suffix = '-rnd'

        configuration_value = None
        if word_evaluation_type == WordEvaluationType.Baseline:
            configuration_value = Configuration.SkipGram

        word_eval_type_suffix = ''
        if word_evaluation_type != WordEvaluationType.Baseline:
            word_eval_type_suffix = f'-{str(word_evaluation_type.value)}'

        if word_evaluation_type != WordEvaluationType.CurrentOriginal:
            word_eval_type_suffix = f'{word_eval_type_suffix}-lr{self._arguments_service.learning_rate}'

        result = CacheOptions(
            'word-similarities',
            seed_specific=True,
            key_suffixes=[
                word_eval_type_suffix,
                '-sep' if self._arguments_service.separate_neighbourhood_vocabularies else '',
                random_suffix,
                '-min',
                str(self._arguments_service.minimal_occurrence_limit)
            ],
            configuration=configuration_value)

        return result

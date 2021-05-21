from services.experiments.process.neighbourhood_overlap_process_service import NeighbourhoodOverlapProcessService
from services.plot_service import PlotService
from services.file_service import FileService
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from typing import Dict, List
from entities.plot.figure_options import FigureOptions
from entities.plot.plot_options import PlotOptions
from enums.experiment_type import ExperimentType


class SetSizedBasedPlotService:
    def __init__(
        self,
        arguments_service: OCREvaluationArgumentsService,
        file_service: FileService,
        plot_service: PlotService,
        neighbourhood_overlap_process_service: NeighbourhoodOverlapProcessService):
        self._file_service = file_service
        self._arguments_service = arguments_service
        self._plot_service = plot_service
        self._neighbourhood_overlap_process_service = neighbourhood_overlap_process_service

    def plot_set_size_bases(self):
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

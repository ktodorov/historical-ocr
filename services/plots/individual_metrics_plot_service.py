from collections import Counter
from services.log_service import LogService
from services.file_service import FileService
from services.plot_service import PlotService
from entities.plot.figure_options import FigureOptions
from entities.plot.legend_options import LegendOptions
from entities.plot.plot_options import PlotOptions
from enums.overlap_type import OverlapType
from enums.experiment_type import ExperimentType
from typing import Dict
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService


class IndividualMetricsPlotService:
    def __init__(
        self,
        arguments_service: OCREvaluationArgumentsService,
        file_service: FileService,
        plot_service: PlotService,
        log_service: LogService):
        self._file_service = file_service
        self._arguments_service = arguments_service
        self._plot_service = plot_service
        self._log_service = log_service

    def plot_individual_metrics(self, result: Dict[ExperimentType, Dict[str, float]]):
        for experiment_type, word_value_pairs_by_overlap in result.items():
            if experiment_type == ExperimentType.NeighbourhoodOverlap:
                for overlap_type, word_value_pairs in word_value_pairs_by_overlap.items():
                    self._save_individual_experiment(
                        experiment_type, overlap_type, word_value_pairs)
            else:
                self._save_individual_experiment(
                    experiment_type, None, word_value_pairs_by_overlap)

    def _save_individual_experiment(
            self,
            experiment_type: ExperimentType,
            overlap_type: OverlapType,
            word_value_pairs):
        experiments_folder = self._file_service.get_experiments_path()
        experiment_type_folder = self._file_service.combine_path(
            experiments_folder,
            experiment_type.value,
            create_if_missing=True)

        if overlap_type is not None:
            experiment_type_folder = self._file_service.combine_path(
                experiment_type_folder,
                overlap_type.value,
                create_if_missing=True)

        self._log_service.log_debug(
            f'Saving \'{experiment_type.value}\' experiment results at \'{experiment_type_folder}\'')

        values = [round(x, 1) for x in word_value_pairs.values()]
        if values is None or len(values) == 0:
            return

        counter = Counter(values)
        filename = f'{self._arguments_service.get_configuration_name()}-{self._arguments_service.neighbourhood_set_size}'
        self._plot_service.plot_distribution(
            counts=counter,
            plot_options=PlotOptions(
                legend_options=LegendOptions(show_legend=False),
                figure_options=FigureOptions(
                    title=experiment_type.value,
                    save_path=experiment_type_folder,
                    filename=filename),
                color='royalblue',
                fill=True,
                xlim=(0, self._arguments_service.neighbourhood_set_size)))

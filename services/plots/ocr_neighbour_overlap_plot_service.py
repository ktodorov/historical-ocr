from collections import defaultdict
from typing import Dict, List
from entities.plot.plot_options import PlotOptions
from enums.plots.line_style import LineStyle
from services.log_service import LogService
from entities.plot.legend_title_options import LegendTitleOptions
from entities.plot.legend_options import LegendOptions
from enums.configuration import Configuration
from entities.plot.figure_options import FigureOptions
from enums.value_summary import ValueSummary
from services.plot_service import PlotService
from services.experiments.process.neighbourhood_overlap_process_service import NeighbourhoodOverlapProcessService
from enums.overlap_type import OverlapType
from enums.experiment_type import ExperimentType
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from services.file_service import FileService
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


class OCRNeighbourOverlapPlotService:
    def __init__(
            self,
            arguments_service: OCREvaluationArgumentsService,
            file_service: FileService,
            plot_service: PlotService,
            log_service: LogService,
            neighbourhood_overlap_process_service: NeighbourhoodOverlapProcessService):
        self._file_service = file_service
        self._arguments_service = arguments_service
        self._plot_service = plot_service
        self._log_service = log_service
        self._neighbourhood_overlap_process_service = neighbourhood_overlap_process_service

    def plot_ocr_ground_truth_overlaps(self, total_words_count: int):
        self._log_service.log_info(
            'Generating OCR vs Ground Truth overlap plots')

        output_folder = self._get_output_folder()
        output_config_folder = self._file_service.combine_path(output_folder, 'configurations', create_if_missing=True)
        overlaps_by_config = self._neighbourhood_overlap_process_service.get_overlaps(
            overlap_types=[OverlapType.GTvsOCR], include_randomly_initialized=True)
        ax = self._plot_service.create_plot()

        for configuration, overlaps_by_type in overlaps_by_config.items():
            for _, overlaps_by_random_initialization in overlaps_by_type.items():
                for randomly_initialized, overlaps_by_lr in overlaps_by_random_initialization.items():
                    for learning_rate, overlaps_by_seed in overlaps_by_lr.items():
                        if all(x is None for x in list(overlaps_by_seed.values())):
                            continue

                        pd_dataframe = self._combine_seed_overlaps(
                            overlaps_by_seed,
                            total_words_count=total_words_count)

                        plot_options = self._get_distribution_plot_options(
                            ax,
                            configuration,
                            randomly_initialized,
                            learning_rate,
                            ValueSummary.Average)

                        ax = self._plot_service.plot_line_variance(
                            pd_dataframe,
                            x='percentage of total data',
                            y='percentage overlap',
                            plot_options=plot_options)

                            # Save the plot also individually
                            # plot_options._ax = None
                            # rnd_initialized_str = 'random' if randomly_initialized else 'pretrained'
                            # title_suffix = f' [{rnd_initialized_str}, LR: {learning_rate}]'
                            # filename_suffix = f'-{rnd_initialized_str}-{learning_rate}'
                            # if configuration == Configuration.PPMI:
                            #     title_suffix = ''
                            #     filename_suffix = ''

                            # plot_options.legend_options._show_legend = False
                            # plot_options._figure_options = FigureOptions(
                            #     title=f'{Configuration.get_friendly_name(configuration)}{title_suffix}',
                            #     save_path=output_config_folder,
                            #     filename=f'{configuration}{filename_suffix}')

                            # self._plot_service.plot_line_variance(
                            #     x_values=list(overlap_line.keys()),
                            #     max_y_values=max_values,
                            #     min_y_values=min_values,
                            #     avg_y_values=avg_values,
                            #     plot_options=plot_options)

            self._plot_service.set_plot_properties(
                ax=ax,
                figure_options=FigureOptions(
                    title=f'Neighbourhood overlaps [GT vs. OCR] ({self._arguments_service.language.value.capitalize()})'))

        self._plot_service.save_plot(
            save_path=output_folder,
            filename=f'neighbourhood-overlaps')

    def _get_output_folder(self):
        experiments_folder = self._file_service.get_experiments_path()
        result = self._file_service.combine_path(
            experiments_folder,
            f'{ExperimentType.NeighbourhoodOverlap.value}-gt-vs-ocr',
            self._arguments_service.language.value,
            create_if_missing=True)

        return result

    def _extract_value_summaries(self, combined_overlaps: Dict[int, List[int]]) -> Dict[ValueSummary, List[int]]:
        value_summaries = {
            ValueSummary.Maximum: {},
            ValueSummary.Average: {},
            ValueSummary.Minimum: {},
        }

        avg_overlaps, min_overlaps, max_overlaps = combined_overlaps
        for percentage in avg_overlaps.keys():
            valid_avg_overlaps = [x for x in avg_overlaps[percentage] if x is not None]
            valid_min_overlaps = [x for x in min_overlaps[percentage] if x is not None]
            valid_max_overlaps = [x for x in max_overlaps[percentage] if x is not None]

            value_summaries[ValueSummary.Minimum][percentage] = (min(valid_avg_overlaps), min(valid_min_overlaps), min(valid_max_overlaps))
            value_summaries[ValueSummary.Maximum][percentage] = (max(valid_avg_overlaps), max(valid_min_overlaps), max(valid_max_overlaps))
            value_summaries[ValueSummary.Average][percentage] = (np.mean(valid_avg_overlaps), np.mean(valid_min_overlaps), np.mean(valid_max_overlaps))

        return value_summaries

    def _combine_seed_overlaps(
        self,
        overlaps_by_seed: Dict[int, Dict[str, int]],
        total_words_count: int) -> Dict[int, List[int]]:
        if all(x is None for x in overlaps_by_seed.values()):
            return None

        overlaps_by_percentage = {}

        for overlaps_by_set_percentage in overlaps_by_seed.values():
            if overlaps_by_set_percentage is None:
                continue

            for percentage, current_overlaps in overlaps_by_set_percentage.items():
                total_words_for_current_percentage = int(total_words_count * (percentage / 100.0))

                overlap_values = list(current_overlaps.values())

                overlap_values = [x[0] if isinstance(x, list) else x for x in overlap_values]
                overlap_values = [(x / total_words_for_current_percentage) * 100 for x in overlap_values]

                if percentage not in overlaps_by_percentage.items():
                    overlaps_by_percentage[percentage] = []

                overlaps_by_percentage[percentage].extend(overlap_values)

        flatten_overlaps = [(percentage, overlap) for percentage, overlaps in overlaps_by_percentage.items() for overlap in overlaps]
        result = pd.DataFrame(flatten_overlaps)
        result.columns = ['percentage of total data', 'percentage overlap']
        return result

    def _get_distribution_plot_options(
            self,
            ax: Axes,
            configuration: Configuration,
            randomly_initialized: bool,
            learning_rate_str: str,
            value_summary: ValueSummary) -> PlotOptions:
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
            Configuration.BERT: {
                ValueSummary.Maximum: 'goldenrod',
                ValueSummary.Average: 'goldenrod',
                ValueSummary.Minimum: 'white',
            },
            Configuration.CBOW: {
                ValueSummary.Maximum: 'cadetblue',
                ValueSummary.Average: 'cadetblue',
                ValueSummary.Minimum: 'white',
            },
            Configuration.SkipGram: {
                ValueSummary.Maximum: 'darkred',
                ValueSummary.Average: 'darkred',
                ValueSummary.Minimum: 'white',
            },
            Configuration.PPMI: {
                ValueSummary.Maximum: 'black',
                ValueSummary.Average: 'black',
                ValueSummary.Minimum: 'white',
            }
        }

        lr_types = {
            f'{Configuration.BERT.value}-0.0001': 'aggressive',
            f'{Configuration.BERT.value}-0.00001': 'slow',
            f'{Configuration.CBOW.value}-0.001': 'aggressive',
            f'{Configuration.CBOW.value}-0.025': 'aggressive',
            f'{Configuration.CBOW.value}-0.0001': 'slow',
            f'{Configuration.SkipGram.value}-0.001': 'aggressive',
            f'{Configuration.SkipGram.value}-0.025': 'aggressive',
            f'{Configuration.SkipGram.value}-0.0001': 'slow',
            f'{Configuration.PPMI.value}': 'aggressive'
        }

        line_styles_per_lr_type = {
            'aggressive': LineStyle.Solid,
            'slow': LineStyle.Dashed
        }

        line_style_key = f'{configuration.value}'
        lr_label = 'default'
        lr_type = 'aggressive'
        if configuration != Configuration.PPMI:
            line_style_key = f'{line_style_key}-{learning_rate_str}'
            lr_type = lr_types[line_style_key]
            lr_label = lr_type

        if randomly_initialized:
            lr_label += ', randomly initialized'

        result = PlotOptions(
            color=colors[configuration][value_summary],
            linestyle=line_styles_per_lr_type[lr_type],
            fill=fill[value_summary],
            label=f'{Configuration.get_friendly_name(configuration)} [{lr_label}]',
            alpha=alpha_values[value_summary],
            line_width=linewidths[value_summary],
            ax=ax,
            ylim=(0, 100),
            xlim=(0, 20),
            legend_options=LegendOptions(show_legend=True))

        return result

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

class BaselineNeighbourOverlapPlotService:
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

    def plot_baseline_overlaps(self):
        self._log_service.log_info('Generating baseline overlap plots')

        output_folder = self._get_output_folder()
        overlaps_by_config = self._neighbourhood_overlap_process_service.get_overlaps(
            self._arguments_service.neighbourhood_set_size,
            overlap_types=[OverlapType.BASEvsGT, OverlapType.BASEvsOCR, OverlapType.BASEvsOG])
        fig, config_axs = self._plot_service.create_plots(
            len(overlaps_by_config.keys()),
            share_x_coords=True)

        for i, (configuration, overlaps_by_type) in enumerate(overlaps_by_config.items()):
            sub_titles = {}
            types_plotted = []
            plot_count = 0
            for overlap_type, overlaps_by_lr in overlaps_by_type.items():
                for learning_rate, overlaps_by_seed in overlaps_by_lr.items():
                    if all(x is None for x in list(overlaps_by_seed.values())):
                        continue

                    if overlap_type not in types_plotted:
                        sub_titles[plot_count] = OverlapType.get_friendly_name(
                            overlap_type)
                        types_plotted.append(overlap_type)
                        plot_count += 1

                    plot_count += 1
                    combined_overlaps = self._neighbourhood_overlap_process_service.combine_seed_overlaps(
                        overlaps_by_seed,
                        self._arguments_service.neighbourhood_set_size)

                    value_summaries = self._neighbourhood_overlap_process_service.extract_value_summaries(
                        combined_overlaps)

                    for value_summary, overlap_line in value_summaries.items():
                        # Skip max and min value summaries
                        if value_summary != ValueSummary.Average:
                            continue

                        config_axs[i] = self._plot_service.plot_distribution(
                            counts=overlap_line,
                            plot_options=self._neighbourhood_overlap_process_service.get_distribution_plot_options(
                                config_axs[i],
                                configuration,
                                overlap_type,
                                learning_rate,
                                value_summary))

            self._plot_service.set_plot_properties(
                ax=config_axs[i],
                figure_options=FigureOptions(
                    hide_y_labels=True,
                    figure=fig,
                    super_title=f'Neighbourhood overlaps [Baseline vs. GT] ({self._arguments_service.language.value.capitalize()})',
                    title=Configuration.get_friendly_name(configuration)),
                legend_options=LegendOptions(
                    show_legend=len(sub_titles) > 0,
                    legend_title_options=LegendTitleOptions(
                        sub_titles=sub_titles)))

        self._plot_service.save_plot(
            save_path=output_folder,
            filename=f'neighbourhood-overlaps-{self._arguments_service.neighbourhood_set_size}',
            figure=fig)

    def _get_output_folder(self):
        experiments_folder = self._file_service.get_experiments_path()
        result = self._file_service.combine_path(
            experiments_folder,
            f'{ExperimentType.NeighbourhoodOverlap.value}-base-vs-gt',
            self._arguments_service.language.value,
            create_if_missing=True)

        return result
from collections import defaultdict
from entities.plot.legend_options import LegendOptions
from entities.plot.label_options import LabelOptions
from enums.overlap_type import OverlapType
from enums.plots.line_style import LineStyle

import numpy as np
from services.cache_service import CacheService
from services.arguments.ocr_evaluation_arguments_service import OCREvaluationArgumentsService
from entities.cache.cache_options import CacheOptions
from typing import Dict, List, Tuple
from entities.plot.plot_options import PlotOptions
from enums.value_summary import ValueSummary
from enums.configuration import Configuration
from matplotlib.axes import Axes


class NeighbourhoodOverlapProcessService:
    def __init__(
        self,
        arguments_service: OCREvaluationArgumentsService,
        cache_service: CacheService):
        self._arguments_service = arguments_service
        self._cache_service = cache_service

    def get_all_overlaps(self, neighbourhood_set_size: int) -> Dict[Tuple[Configuration, bool], Dict[int, dict]]:
        configurations = [
            Configuration.CBOW,
            Configuration.PPMI,
            Configuration.SkipGram,
            Configuration.BERT,
        ]

        seeds = [7, 13, 42]

        result = {}

        for configuration in configurations:
            for is_random_initialized in [False, True]:
                result[(configuration, is_random_initialized)] = {}

                for seed in seeds:
                    config_overlaps = self._cache_service.get_item_from_cache(
                        CacheOptions(
                            'neighbourhood-overlaps',
                            key_suffixes=[
                                '-rnd' if is_random_initialized else '',
                                f'-{neighbourhood_set_size}'
                            ],
                            configuration=configuration,
                            seed=seed))

                    result[(configuration, is_random_initialized)][seed] = config_overlaps

        return result

    def get_overlaps(self, neighbourhood_set_size: int) -> Dict[Configuration, Dict[str, Dict[OverlapType, Dict[int, dict]]]]:
        seeds = [7, 13, 42]
        lrs = ['0.01', '0.001', '0.0001', '0.00001']
        configs = [Configuration.BERT, Configuration.SkipGram, Configuration.CBOW, Configuration.PPMI]

        result = {}

        for config in configs:
            result[config] = {}
            for overlap_type in OverlapType:
                if overlap_type == OverlapType.GTvsOCR:
                    continue

                result[config][overlap_type] = {}
                for lr in lrs:
                    result[config][overlap_type][lr] = {}

                    for seed in seeds:
                        overlaps = self._cache_service.get_item_from_cache(
                            CacheOptions(
                                'neighbourhood-overlaps',
                                key_suffixes=[
                                    '-lr' if config != Configuration.PPMI else '',
                                    lr if config != Configuration.PPMI else '',
                                    '-',
                                    str(overlap_type.value),
                                    '-',
                                    str(neighbourhood_set_size)
                                ],
                                configuration=config,
                                seed=seed,
                                seed_specific=True))

                        result[config][overlap_type][lr][seed] = overlaps

                    # If we are processing PPMI, we only have one LR so we break
                    if config == Configuration.PPMI:
                        break

        return result

    def combine_seed_overlaps(
        self,
        overlaps_by_seed: Dict[int, Dict[str, int]],
        neighbourhood_set_size: int,
        max_bins: int = 100) -> Dict[int, List[int]]:
        if all(x is None for x in overlaps_by_seed.values()):
            return None

        reduce_factor = 1
        if neighbourhood_set_size > max_bins:
            reduce_factor = neighbourhood_set_size / max_bins

        combined_overlaps = defaultdict(
            lambda: [None for _ in range(len(overlaps_by_seed.keys()))])

        for i, current_overlaps in enumerate(overlaps_by_seed.values()):
            if current_overlaps is None:
                continue

            for overlap_amount in current_overlaps.values():
                overlap_amount = int(overlap_amount / reduce_factor)

                if combined_overlaps[overlap_amount][i] is None:
                    combined_overlaps[overlap_amount][i] = 0

                combined_overlaps[overlap_amount][i] = combined_overlaps[overlap_amount][i] + 1

        return combined_overlaps

    def extract_value_summaries(self, combined_overlaps: Dict[int, List[int]]) -> Dict[ValueSummary, List[int]]:
        value_summaries = {
            ValueSummary.Maximum: {},
            ValueSummary.Average: {},
            ValueSummary.Minimum: {},
        }

        for i in range(0, self._arguments_service.neighbourhood_set_size, 1):
            if i not in combined_overlaps.keys():
                continue

            valid_overlaps = [x for x in combined_overlaps[i] if x is not None]

            value_summaries[ValueSummary.Minimum][i] = min(valid_overlaps)
            value_summaries[ValueSummary.Maximum][i] = max(valid_overlaps)
            value_summaries[ValueSummary.Average][i] = int(np.mean(valid_overlaps))

        return value_summaries

    def get_distribution_plot_options(
        self,
        ax: Axes,
        configuration: Configuration,
        overlap_type: OverlapType,
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
            OverlapType.BASEvsGT: {
                ValueSummary.Maximum: 'goldenrod',
                ValueSummary.Average: 'goldenrod',
                ValueSummary.Minimum: 'white',
            },
            OverlapType.BASEvsOG: {
                ValueSummary.Maximum: 'cadetblue',
                ValueSummary.Average: 'cadetblue',
                ValueSummary.Minimum: 'white',
            },
            OverlapType.BASEvsOCR: {
                ValueSummary.Maximum: 'darkred',
                ValueSummary.Average: 'darkred',
                ValueSummary.Minimum: 'white',
            }
        }

        lr_types = {
            f'{Configuration.BERT.value}-0.0001': 'aggressive',
            f'{Configuration.BERT.value}-0.00001': 'slow',
            f'{Configuration.CBOW.value}-0.001': 'aggressive',
            f'{Configuration.CBOW.value}-0.0001': 'slow',
            f'{Configuration.SkipGram.value}-0.001': 'aggressive',
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


        result = PlotOptions(
            color=colors[overlap_type][value_summary],
            linestyle=line_styles_per_lr_type[lr_type],
            fill=fill[value_summary],
            label=lr_label,
            alpha=alpha_values[value_summary],
            line_width=linewidths[value_summary],
            ax=ax,
            xlim=(0,100),
            legend_options=LegendOptions(show_legend=False))

        return result
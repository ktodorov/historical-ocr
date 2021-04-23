from collections import defaultdict
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

    def get_main_overlaps(self, neighbourhood_set_size: int) -> Dict[Tuple[Configuration], Dict[int, dict]]:
        configurations = [
            Configuration.CBOW,
            Configuration.PPMI,
            Configuration.SkipGram,
            Configuration.BERT,
        ]

        seeds = [7, 13, 42]

        result = {}

        for configuration in configurations:
            result[configuration] = {}

            for seed in seeds:
                config_overlaps = self._cache_service.get_item_from_cache(
                    CacheOptions(
                        'neighbourhood-overlaps-vs-base',
                        key_suffixes=[
                            f'-{neighbourhood_set_size}'
                        ],
                        configuration=configuration,
                        seed=seed))

                result[configuration][seed] = config_overlaps

        return result

    def get_base_overlaps(self, neighbourhood_set_size: int) -> Dict[int, dict]:
        seeds = [7, 13, 42]
        configuration = Configuration.SkipGram
        result = {}

        for seed in seeds:
            config_overlaps = self._cache_service.get_item_from_cache(
                CacheOptions(
                    'neighbourhood-overlaps',
                    key_suffixes=[
                        '-rnd',
                        f'-{neighbourhood_set_size}'
                    ],
                    configuration=configuration,
                    seed=seed))

            result[seed] = config_overlaps

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
        is_random_initialized: bool,
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
            False: {
                Configuration.CBOW: {
                    ValueSummary.Maximum: 'darkgoldenrod',
                    ValueSummary.Average: 'darkgoldenrod',
                    ValueSummary.Minimum: 'white',
                },
                Configuration.PPMI: {
                    ValueSummary.Maximum: 'green',
                    ValueSummary.Average: 'green',
                    ValueSummary.Minimum: 'white',
                },
                Configuration.SkipGram: {
                    ValueSummary.Maximum: 'darkblue',
                    ValueSummary.Average: 'darkblue',
                    ValueSummary.Minimum: 'white',
                },
                Configuration.BERT: {
                    ValueSummary.Maximum: 'crimson',
                    ValueSummary.Average: 'crimson',
                    ValueSummary.Minimum: 'white',
                }
            },
            True: {
                Configuration.CBOW: {
                    ValueSummary.Maximum: 'orange',
                    ValueSummary.Average: 'orange',
                    ValueSummary.Minimum: 'white',
                },
                Configuration.PPMI: {
                    ValueSummary.Maximum: 'seagreen',
                    ValueSummary.Average: 'seagreen',
                    ValueSummary.Minimum: 'white',
                },
                Configuration.SkipGram: {
                    ValueSummary.Maximum: 'royalblue',
                    ValueSummary.Average: 'royalblue',
                    ValueSummary.Minimum: 'white',
                },
                Configuration.BERT: {
                    ValueSummary.Maximum: 'palevioletred',
                    ValueSummary.Average: 'palevioletred',
                    ValueSummary.Minimum: 'white',
                },
            }
        }

        line_styles = {
            ValueSummary.Maximum: LineStyle.Solid,
            ValueSummary.Average: LineStyle.Dashed,
            ValueSummary.Minimum: LineStyle.Solid,
        }

        random_label_suffix = ' [random]' if is_random_initialized else ''
        result = PlotOptions(
            color=colors[is_random_initialized][configuration][value_summary],
            linestyle=line_styles[value_summary],
            fill=fill[value_summary],
            label=f'{configuration.value} [{value_summary.value}]{random_label_suffix}',
            alpha=alpha_values[value_summary],
            line_width=linewidths[value_summary],
            ax=ax)

        return result
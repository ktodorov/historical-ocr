from enums.plot_legend_position import PlotLegendPosition
from typing import List


class LegendOptions:
    def __init__(
            self,
            show_legend: bool,
            legend_colors: List[str],
            legend_labels: List[str],
            legend_position: PlotLegendPosition = PlotLegendPosition.Automatic):
        self._show_legend = show_legend
        self._legend_colors: List[str] = legend_colors
        self._legend_labels: List[str] = legend_labels
        self._legend_position = legend_position

    @property
    def show_legend(self) -> bool:
        return self._show_legend

    @property
    def legend_colors(self) -> List[str]:
        return self._legend_colors

    @property
    def legend_labels(self) -> List[str]:
        return self._legend_labels

    @property
    def legend_position(self) -> PlotLegendPosition:
        return self._legend_position
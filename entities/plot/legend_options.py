from typing import List


class LegendOptions:
    def __init__(
            self,
            show_legend: bool,
            legend_colors: List[str],
            legend_labels: List[str]):
        self._show_legend = show_legend
        self._legend_colors: List[str] = legend_colors
        self._legend_labels: List[str] = legend_labels

    @property
    def show_legend(self) -> bool:
        return self._show_legend

    @property
    def legend_colors(self) -> List[str]:
        return self._legend_colors

    @property
    def legend_labels(self) -> List[str]:
        return self._legend_labels

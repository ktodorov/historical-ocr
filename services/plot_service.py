from entities.plot.legend_options import LegendOptions
import sys
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from services.data_service import DataService
from typing import List
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from collections import Counter

plt.rcParams["figure.figsize"] = (15, 10)
# plt.rcParams["text.usetex"] = True
# plt.rcParams['text.latex.preamble'] = [
#     r'\usepackage[utf8]{inputenc}'
#     r'\usepackage[cm]{sfmath}'
# ]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
# plt.rcParams['text.latex.preamble'] = [
#        r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#        r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#        r'\usepackage{helvet}',    # set the normal font here
#        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
# ]

class PlotService:
    def __init__(
            self,
            data_service: DataService):
        sns.set()
        # sns.set(font_scale=2)  # crazy big
        sns.set_style("ticks")

        self._data_service = data_service

        self._default_font_size: int = 10

    def create_plot(self) -> Axes:
        fig = plt.figure()
        fig.canvas.start_event_loop(sys.float_info.min) #workaround for Exception in Tkinter callback
        ax = fig.add_subplot(1,1,1)
        # ax = plt.subplot()
        return ax

    def plot_histogram(
            self,
            values: list,
            number_of_bins: int = None,
            start_x: float = None,
            end_x: float = None,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            ax=None,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        if number_of_bins is None:
            number_of_bins = len(set(values))

        if not start_x:
            start_x = min(values)

        if not end_x:
            end_x = max(values)

        distance_bin = (end_x - start_x) / number_of_bins

        bins = np.arange(start_x, end_x, distance_bin)

        ax.hist(values, bins=bins, edgecolor='none')

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis)

        if save_path is None or filename is None:
            plt.show()

        plt.clf()

        return ax

    def autolabel_heights(self, ax, rects, rotation: int = 0):
        """Attach a text label above each bar in *rects*, displaying its height."""
        y_offset = 3 if rotation == 0 else 10
        for rect in rects:
            height = rect.get_height()
            if height == 0:
                continue

            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, y_offset),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=rotation)

    def plot_counters_histogram(
            self,
            counter_labels: List[str],
            counters: List[Counter],
            counter_colors: List[str] = None,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            xlabel: str = None,
            ylabel: str = None,
            plot_values_above_bars: bool = False,
            values_above_bars_rotation: int = 0,
            x_labels_rotation_angle: int = 0,
            space_x_labels_vertically: bool = False,
            tight_layout: bool = True,
            external_legend: bool = False,
            show_legend: bool = True,
            ax=None,
            bars_padding: float = 0.2,
            xticks_count: int = None,
            hide_axis: bool = False):

        if ax is None:
            ax = self.create_plot()

        unique_labels = list(
            sorted(set([label for x in counters for label in x.keys()])))

        values = []
        for counter in counters:
            values.append([(counter[label] if label in counter.keys() else 0)
                           for label in unique_labels])

        if bars_padding >= 1 or bars_padding < 0:
            raise Exception('bars_padding must be between 0 and 1')

        total_width = 1 - bars_padding  # the width of the bars
        dim = len(counters)
        dimw = total_width / dim

        x = np.arange(len(unique_labels))  # the label locations

        if counter_colors is None:
            counter_colors = cm.rainbow(np.linspace(0, 1, dim))

        rects = []
        for i, counter_values in enumerate(values):
            rects.append(
                ax.bar(x + (i * dimw), counter_values, dimw, label=counter_labels[i], color=counter_colors[i]))

        xticks = x + (total_width - dimw) / 2
        xtick_labels = unique_labels
        if xticks_count is not None:

            indices = np.round(np.linspace(0, len(xticks) - 1, xticks_count)).astype(int)
            leftover_ticks = [xticks[idx] for idx in indices]
            xtick_labels = [unique_labels[idx] for idx in indices]

            xticks = leftover_ticks

        ax.set_xticks(xticks)
        labels = ax.set_xticklabels(
            xtick_labels, rotation=x_labels_rotation_angle)

        if space_x_labels_vertically:
            for i, label in enumerate(labels):
                label.set_y(label.get_position()[1] - (i % 2) * 0.075)

        if show_legend:
            if external_legend:
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                ax.legend()

        # fontweight = 'bold'
        # fontproperties = {
        #     'family': 'sans-serif',
        #     'sans-serif': ['Helvetica'],
        #     'weight': fontweight
        # }

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontweight='bold')

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontweight='bold')

        if plot_values_above_bars:
            for rect in rects:
                self.autolabel_heights(
                    ax, rect, rotation=values_above_bars_rotation)

        x1, x2, y1, y2 = ax.axis()
        ax.axis((x1, x2, y1, y2 + 5))

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis,
            tight_layout)

        if save_path is None or filename is None:
            plt.show()

        plt.clf()

        return ax

    def plot_scatter(
            self,
            x_values: list,
            y_values: list,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            color: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        ax.scatter(x_values, y_values, color=color)

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

        return ax

    def plot_overlapping_bars(
            self,
            numbers_per_type: List[List[int]],
            bar_titles: List[str],
            colors: List[str] = None,
            show_legend: bool = False,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False,
            tight_layout: bool = False,
            ylim: float = None,
            xlim: float = None,
            ylabel: str = None,
            xlabel: str = None):

        if ax is None:
            ax = self.create_plot()

        unique_numbers = set([item for v in numbers_per_type for item in v])
        counters_per_type = {bar_titles[i]: Counter(
            v) for i, v in enumerate(numbers_per_type)}

        normalized_counters_per_type = {
            type_name: Counter({
                n: (float(v)/sum(unnormalized_counter.values())) * 100
                for n, v in unnormalized_counter.items()
            })
            for type_name, unnormalized_counter in counters_per_type.items()
        }

        # print(normalized_counters_per_type)

        argmaxes = {}
        for i, number in enumerate(unique_numbers):
            occs = np.array([x[number]
                             for _, x in normalized_counters_per_type.items()])
            arg_sort = np.argsort(np.argsort(occs, kind='heapsort'))
            sorted_occs = sorted(occs)

            a = np.zeros(len(occs))
            for i, index in enumerate(arg_sort):
                if index == 0:
                    a[i] = 0
                else:
                    a[i] = sorted_occs[index-1]

            argmaxes[number] = a

        for i, counter_values in enumerate(normalized_counters_per_type.values()):
            x = list(sorted(counter_values.keys()))

            y = np.array([counter_values[key] for key in x])
            p = np.array([argmaxes[a][i] for a in x])
            norm_y = y - p

            ax.bar(x, norm_y, width=x[1]-x[0], color=colors[i], bottom=p)

        if ylim is not None:
            ax.set_ylim(top=ylim)

        if xlim is not None:
            ax.set_xlim(right=xlim)

        # fontweight = 'bold'
        # fontproperties = {
        #     'family': 'sans-serif',
        #     'sans-serif': ['Helvetica'],
        #     'weight': fontweight
        # }

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if show_legend:
            ax.legend(bar_titles)

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis,
            tight_layout)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

        return ax

    def plot_labels(
            self,
            x_values: list,
            y_values: list,
            labels: list,
            color: str = None,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            ax=None,
            show_plot: bool = True,
            bold_mask: list = None,
            hide_axis: bool = False,
            font_sizes: List[int] = None):
        if ax is None:
            ax = self.create_plot()

        if font_sizes is None or len(font_sizes) < len(labels):
            font_sizes = [self._default_font_size for _ in range(len(labels))]
        else:
            font_sizes = [font_sizes[i] if font_sizes[i] is not None else self._default_font_size for i in range(len(font_sizes))]

        for i, (label, x, y) in enumerate(zip(labels, x_values, y_values)):
            weight = 'light'
            if bold_mask is not None and bold_mask[i]:
                weight = 'bold'

            ax.annotate(label, xy=(x, y), xytext=(
                0, 0), textcoords='offset points', color=color, weight=weight, font_size=font_sizes[i])

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

        return ax

    def plot_arrow(
            self,
            x: float,
            y: float,
            dx: float,
            dy: float,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            color: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color),
                    bbox=dict(pad=7, facecolor="none", edgecolor="none"))

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

        return ax

    def plot_confusion_matrix(
            self,
            true_values: list,
            predicted_values: list,
            labels: List[str] = None,
            normalize: bool = False,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        cm = confusion_matrix(true_values, predicted_values, labels)

        vmin = cm.min()
        vmax = cm.max()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            vmin = 0
            vmax = 1

        sns_heatmap = sns.heatmap(
            cm,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap='RdYlGn_r',
            square=True)

        ax.set_xlabel('Predicted values')  # , labelpad=20)
        ax.set_ylabel('True values')

        if labels is not None:
            ax.set_ylim(0, len(labels) + 0.5)
            ax.set_ylim(0, len(labels) + 0.5)

            sns_heatmap.set_yticklabels(labels, rotation=0)
            sns_heatmap.set_xticklabels(
                labels, rotation=45, horizontalalignment='right')

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

        return ax

    def plot_heatmap(
            self,
            values: np.array,
            labels: List[str] = None,
            title: str = None,
            title_padding: float = None,
            vmin: float = None,
            vmax: float = None,
            y_title: str = None,
            x_title: str = None,
            show_colorbar: bool = True,
            save_path: str = None,
            filename: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        if vmin is None:
            vmin = np.min(values)

        if vmax is None:
            vmax = np.max(values)

        sns_heatmap = sns.heatmap(
            values,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap='Greens',
            square=True,
            cbar=show_colorbar)

        if x_title is not None:
            ax.set_xlabel(x_title)

        if y_title is not None:
            ax.set_ylabel(y_title)

        if labels is not None:
            ax.set_ylim(0, len(labels) + 0.5)
            ax.set_ylim(0, len(labels) + 0.5)

            sns_heatmap.set_yticklabels(labels, rotation=0)
            sns_heatmap.set_xticklabels(
                labels, rotation=45, horizontalalignment='right')

        self._add_properties(
            ax,
            title,
            title_padding,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

        return ax

    def show_plot(self):
        plt.show()

    def set_plot_properties(
        self,
        ax: Axes,
        title: str = None,
        title_padding: float = None,
        hide_axis: bool = False,
        tight_layout: bool = True,
        legend_options: LegendOptions = None):

        if tight_layout:
            plt.tight_layout()

        if hide_axis:
            ax.axis('off')

        if legend_options is not None:
            self.show_legend(ax, legend_options)

        if title is not None:
            ax.set_title(title, pad=title_padding,
                         fontdict={'fontweight': 'bold'})


    def show_legend(
        self,
        ax: Axes,
        legend_options: LegendOptions):

        if legend_options is None or not legend_options.show_legend:
            return

        if legend_options.legend_colors is not None and len(legend_options.legend_colors) > 0:
            legend_lines = self._create_legend_lines(legend_options.legend_colors)
            if legend_options.legend_labels is not None and len(legend_options.legend_labels) > 0:
                ax.legend(legend_lines, legend_options.legend_labels)
            else:
                ax.legend(legend_lines)
        else:
            ax.legend()


    def save_plot(
        self,
        save_path: str,
        filename: str):
        self._data_service.save_figure(save_path, filename, no_axis=False)

    def _create_legend_lines(
        self,
        legend_colors: List[str]) -> List[Artist]:
        lines = [Line2D([0], [0], color=color, lw=4) for color in legend_colors]
        return lines


    def _add_properties(
            self,
            ax: Axes,
            title: str = None,
            title_padding: float = None,
            save_path: str = None,
            filename: str = None,
            hide_axis: bool = False,
            tight_layout: bool = True):

        self.set_plot_properties(
            ax,
            title,
            title_padding,
            hide_axis,
            tight_layout)

        if save_path is not None and filename is not None:
            self.save_plot(save_path, filename)
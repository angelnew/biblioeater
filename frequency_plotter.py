import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import random
import numpy as np


class FrequencyPlotter:

    def __init__(self, plot_title=""):

        self.fig = plt.figure(figsize=(12.8, 8))
        self.ax = self.fig.add_subplot(1,1,1)

        # custom frame
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_linewidth(1.5)
        self.ax.spines['left'].set_linewidth(1.5)

        # grid
        kwargs = {}
        plt.grid(b=True, which='major', axis='y', **kwargs)

        # title
        title_properties = FontProperties(family=["Roboto", "sans-serif"], style='normal',
                                                  variant='normal', weight='black', stretch='normal', size='xx-large')
        kwargs = {"fontproperties": title_properties}
        plt.title(plot_title, y=1.05, loc='left', **kwargs)

        # xaxis
        self.upos_xaxis_properties = FontProperties(family=["Roboto", "sans-serif"], style='normal',
                                                  variant='normal', weight='medium', stretch='normal', size='large')
        self.xaxis_properties = FontProperties(family=["Roboto", "sans-serif"], style='normal',
                                                  variant='normal', weight='medium', stretch='normal', size='x-large')

        # colors
        self.colors = ["#3a89c2", "#99bf91"]

    def categorical_hist(self, s1, s2=None, bar_width=0.3, bar_shift=0.7):
        # transform counter
        total = sum(list(s1.values()))
        c1 = {k: v/total for k, v in s1.items()}
        c1 = ordered_counter(c1)

        series_length = len(c1)

        r1 = np.arange(series_length)

        self.ax.bar(r1, list(c1.values()), width=bar_width, color=self.colors[0])

        if s2 is not None:
            total = sum(list(s2.values()))
            c2 = {k: v / total for k, v in s2.items()}
            c2 = ordered_counter(c2)
            series_length = len(c2)
            r1 = np.arange(series_length)
            r2 = [x + bar_width*bar_shift for x in r1]

            self.ax.bar(r2, list(c2.values()), width=bar_width, color=self.colors[1])

        # x axis
        plt.xticks([r - (2*bar_width - bar_shift)/2 for r in range(series_length)], list(c1.keys()))

        self.tune_axis(self.upos_xaxis_properties, 60)
        self.ax.legend(['Arthur Gordon Pym', 'Tom Sawyer'], fontsize="xx-large", facecolor="ghostwhite", framealpha=0.9)

        # plt.show()

    def continuous_hist(self, series1, series2=None, num_bins=25, bar_width=0.3):

        if series2 is None:
            self.ax.hist(series1, bins="auto", bar_width=bar_width)
        else:
            series_1_and_two = (series1, series2)  # np.column_stack((series1, series2))
            self.ax.hist(series_1_and_two, bins="scott", histtype='bar', density=True, color=self.colors)

        self.tune_axis(self.xaxis_properties)
        self.ax.legend(['Arthur Gordon Pym', 'Tom Sawyer'], fontsize="xx-large", facecolor="ghostwhite", framealpha=0.9)
        # plt.show()

    def tune_axis(self, xaxis_props, rotation=0):
        # Fonts for x axis markers
        plt.gcf().subplots_adjust(bottom=0.15)
        for label in self.ax.get_xticklabels():
            label.set_fontproperties(xaxis_props)
            label.set_rotation(rotation)

        # Fonts for y axis markers
        for label in self.ax.get_yticklabels():
            label.set_fontproperties(self.xaxis_properties)

        # percentages in y axis
        y_values = self.ax.get_yticks()
        self.ax.set_yticklabels(['{:,.2%}'.format(y) for y in y_values])

    def persist(self, filename):
        plt.savefig(filename)


def ordered_counter(c):
    return {k: c[k] for k in sorted(c.keys())}


if __name__ == "__main__":
    f1 = [random.randint(10, 200) for i in range(200)]
    f2 = [random.normalvariate(100, 30) for i in range(300)]

    my_plotter = FrequencyPlotter("Some title")
    my_plotter.continuous_hist(f1, f2)

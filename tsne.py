#!/usr/bin/env python3

import argparse
import os
import pickle as p
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.tools import DEFAULT_PLOTLY_COLORS
from plotly.io import to_html
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


def get_args():
    parser = argparse.ArgumentParser(description="Tsne plot from pandas dataframe")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input file (pickle dump of pandas DF)",
    )
    parser.add_argument(
        "-b",
        "--blast",
        type=str,
        required=True,
        help="processed blast results file (csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        required=False,
        default=".",
        help="output directory",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        nargs="?",
        required=False,
        default="output",
        help="name for output files and graph title",
    )
    parser.add_argument(
        "-p",
        "--perplexity",
        type=int,
        nargs="?",
        required=False,
        default=None,
        help="name for output files and graph title",
    )
    return parser.parse_args()


def confidence_ellipse(x, y, n_std=1.96, size=100):
    """
    Get the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    size : int
        Number of points defining the ellipse
    Returns
    -------
    String containing an SVG path for the ellipse

    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5

    from: https://gist.github.com/dpfoose/38ca2f5aee2aea175ecc6e599ca6e973
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)]
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = (
        ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix
    )

    path = f"M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}"
    for k in range(1, len(ellipse_coords)):
        path += f"L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}"
    path += " Z"
    return path


def sort_key(string):
    match = re.search("[0-9]", string)
    return int(string[match.start() :])


def tsne_plot(df, name, perplexity):
    plot = go.Figure()

    for wga in df["wga"].unique():
        wga_df = df[df["wga"] == wga]

        plot.add_trace(
            go.Scatter(
                x=wga_df["tsne-2d-one"],
                y=wga_df["tsne-2d-two"],
                mode="markers",
                name=wga,
                text=wga_df["length"],
                customdata=np.stack(
                    (
                        wga_df["assignment"],
                        wga_df["length"],
                        wga_df["phage count"],
                        wga_df["human count"],
                        wga_df["term-int-terl count"],
                    ),
                    axis=-1,
                ),
                hovertemplate="Assignment: <b>%{customdata[0]}</b><br><br>"
                + "Length: %{customdata[1]}<br>"
                + "Phage count: %{customdata[2]}<br>"
                + "Human count: %{customdata[3]}<br>"
                + "term-int-terl count: %{customdata[4]}",
                marker=dict(
                    symbol=[
                        "circle-open" if x == "bacteria" else "circle"
                        for x in wga_df["assignment"]
                    ],
                    size=[
                        7 if x == "bacteria" else 12
                        for x in wga_df["assignment"]
                    ]
                ),
            )
        )

    color = iter(DEFAULT_PLOTLY_COLORS * 5)

    for cls in df["cluster"].unique():
        this_color = next(color)
        plot.add_shape(
            type="path",
            path=confidence_ellipse(
                x=df[df["cluster"] == cls]["tsne-2d-one"],
                y=df[df["cluster"] == cls]["tsne-2d-two"],
                n_std=1,
                size=100,
            ),
            line={"dash": "longdash", "width": 0.5},
            line_color=this_color,
        )
        plot.add_shape(
            type="path",
            path=confidence_ellipse(
                x=df[df["cluster"] == cls]["tsne-2d-one"],
                y=df[df["cluster"] == cls]["tsne-2d-two"],
                n_std=1.96,
                size=100,
            ),
            line={"dash": "longdash", "width": 0.25},
            line_color=this_color,
        )

    plot.update_layout(title=f"{name} - t-SNE plot<br>perplexity: {perplexity}")

    return plot


def tsne(df, output_dir, name, perplexity):
    reduce = False

    if reduce:
        df = df[
            df["wga"].isin(
                [
                    "WGA09",
                    "WGA10",
                    "WGA11",
                    "WGA12",
                ]
            )
        ]

    descriptor_cols = [
        "index",
        "assignment",
        "length",
        "phage count",
        "human count",
        "term-int-terl count",
        "total matches",
        "wga",
    ]

    to_embed = df[[x for x in df.columns if x not in descriptor_cols]]

    symbol_dict = {"bacteria": "asterisk", "term-int-terl": "star", "phage": "circle"}

    X_embedded = TSNE(
        n_components=2,  # number of dimensions in plot -kate
        learning_rate="auto",
        # method="exact",
        init="random",
        perplexity=perplexity,
    ).fit_transform(to_embed)

    df["tsne-2d-one"] = X_embedded[:, 0]
    df["tsne-2d-two"] = X_embedded[:, 1]
    df.sort_values(by="assignment", inplace=True, ascending=False)

    clustering = DBSCAN(eps=3, min_samples=1).fit(
        df[["tsne-2d-one", "tsne-2d-two"]].to_numpy()
    )
    df["cluster"] = clustering.labels_

    df.to_csv(
        Path(output_dir) / f"{name}_tsne_embeddings.csv", lineterminator="\n", sep=","
    )

    plot = tsne_plot(df, name, perplexity)
    # plot.update_layout(
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     plot_bgcolor='rgba(0,0,0,0)'
    # )
    plot.show()

    with open(os.path.join(output_dir, f"{name}_tnse-plot.html"), "w") as plot_file:
        plot_file.write(to_html(plot, include_plotlyjs="cdn"))


def main():
    args = get_args()
    print(args.__dict__)

    with open(args.input, "rb") as infile:
        df_load = p.load(infile)

    df_assignments = pd.read_csv(args.blast)
    df_assignments["contig"] = df_assignments["contig name"].apply(
        lambda x: x.split(" ")[0]
    )

    df_load.reset_index(inplace=True)
    df = df_load.merge(
        df_assignments, left_on=["index"], right_on=["contig"], how="left"
    )

    df.index = df["index"]
    col_list = list(df_load.columns) + [
        "assignment",
        "length",
        "phage count",
        "human count",
        "term-int-terl count",
        "total matches",
        "wga",
    ]
    df = df[col_list]

    df["wga"] = df["wga"].replace("WGA9", "WGA09")

    # print(df.head())
    perplexity = args.perplexity
    if perplexity is None:
        perplexity = int(round(np.sqrt(len(df)), 0))

    tsne(df, args.output, args.name, perplexity)


if __name__ == "__main__":
    sys.exit(main())

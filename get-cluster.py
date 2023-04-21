import argparse
import re
import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.tools
from sklearn.cluster import DBSCAN


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


def get_args():
    parser = argparse.ArgumentParser(description="Tsne plot from pandas dataframe")
    parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        required=True,
        help="embeddings output from tsne.py",
    )
    parser.add_argument(
        "-b",
        "--blast",
        type=str,
        required=True,
        help="blast results file (blast.out)",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        help="contig name from fasta file (exact match)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    df = pd.read_csv(args.embeddings)[
        [
            "index",
            "wga",
            "length",
            "assignment",
            "phage count",
            "human count",
            "term-int-terl count",
            "tsne-2d-one",
            "tsne-2d-two",
        ]
    ]
    array = df[["tsne-2d-one", "tsne-2d-two"]].to_numpy()

    clustering = DBSCAN(eps=3, min_samples=1).fit(array)
    df["cluster"] = clustering.labels_

    target_cluster = df[df["index"] == args.target]["cluster"].values[0]

    cluster = df[df["cluster"] == target_cluster]

    with open(args.blast, "r") as blast:
        blast = blast.read()

    blast_map = {}
    for row in cluster.itertuples():
        start = re.search(f"Query= {row.index}", blast).start()
        end_match = re.search(">", blast[start:])
        if end_match is None:
            end = 0
        else:
            end = end_match.start() + start

        blast_string = blast[start:end]
        blast_map[row.Index] = blast_string

    cluster["blast"] = pd.Series(blast_map)
    print(cluster)

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
            )
        )

    color = iter(plotly.tools.DEFAULT_PLOTLY_COLORS)

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
            line={"dash": "dash", "width": 1},
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
            line={"dash": "longdash", "width": 0.5},
            line_color=this_color,
        )

    plot.show()

    # with open(os.path.join(output_dir, f"{name}_tnse-plot.html"), "w") as plot_file:
    #     plot_file.write(to_html(plot, include_plotlyjs="cdn"))


if __name__ == "__main__":
    sys.exit(main())

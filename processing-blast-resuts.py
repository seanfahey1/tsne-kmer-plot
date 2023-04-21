#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io


def get_args():
    parser = argparse.ArgumentParser(description="Processing blast output file")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input file from blast output",
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
    return parser.parse_args()


class ContigValueStore:
    phage_terms = ["phage", "siphovir", "myovir", "podovir", "caudovir"]
    prophage_terms = [
        "terminase",
        # "terl", # I'm actually never seeing terl matches
        "integrase"
    ]
    human_terms = ["human", "homosap"]

    def __init__(self, header, blast_text, full_text):
        self.name = header
        self.blast_text = blast_text
        self.blast_text_lower = self.blast_text.lower()

        self.phage_matches = [
            re.findall(term, self.blast_text_lower) for term in self.phage_terms
        ]
        self.human_matches = [
            re.findall(term, self.blast_text_lower) for term in self.human_terms
        ]
        self.prophage_matches = [
            re.findall(term, self.blast_text_lower) for term in self.prophage_terms
        ]

        self.phage_count = sum(len(x) for x in self.phage_matches)
        self.human_count = sum(len(x) for x in self.human_matches)
        self.prophage_count = sum(len(x) for x in self.prophage_matches)

        self.total_count = len(blast_text.split("\n"))
        self.phage_fraction = self.phage_count / self.total_count

        self.wga = re.match("WGA[0-9]*", header).group(0)

        length = re.search("Length=[0-9]*", full_text).group(0)
        self.length = int(re.sub("Length=", "", length))


def read_file(filename):
    with open(filename, "r") as file:
        file = file.read().split("Query= ")[1:]
    file = [x.split("\n\n>")[0] for x in file]

    # TODO: Need to add some logic to truncate the blocks if no matches found,

    df = pd.DataFrame(
        columns=[
            "contig name",
            "wga",
            "length",
            "phage count",
            "human count",
            "term-int-terl count",
            "total matches",
            "phage fraction",
            "phage fraction formatted",
        ]
    )

    for chunk in file:
        lines = chunk.split("\n")

        header = lines[0]
        lines = lines[1:]

        lines = [x for x in lines if not re.fullmatch("[\n\t\b ]*", x)]
        lines = [x for x in lines if not re.search("Score[\t ]*E", x)]
        lines = [x for x in lines if not re.match("Length=[0-9]*", x)]
        lines = [x for x in lines if not re.match("Sequences producing", x)]

        contig = ContigValueStore(header, "\n".join(lines), chunk)

        df.loc[len(df)] = [
            contig.name,
            contig.wga,
            contig.length,
            contig.phage_count,
            contig.human_count,
            contig.prophage_count,
            contig.total_count,
            contig.phage_fraction,
            "{0:.2%}".format(contig.phage_fraction),
        ]

    # df = df.sort_values(by="phage count", ascending=False)

    return df


def map_assignment(row):
    if row['term-int-terl count'] > 0:
        return "term-int-terl"
    elif row['phage fraction'] > 0:
        return "phage"
    else:
        return "bacteria"


def main(plot=True):
    args = get_args()
    print(args.__dict__)

    df = read_file(filename=args.input)

    title = "% of phage terms"

    if plot:
        fig = px.strip(
            df,
            y="phage fraction",
            title=title,
            color="wga",
            hover_name="contig name",
            hover_data=[
                "phage fraction formatted",
                "wga",
                "length",
                "phage count",
                "term-int-terl count",
                "human count",
                "total matches",
            ],
        )
        fig.show()

        with open(Path(args.output)/f'{title}-plot.html', 'w') as out:
            out.write(plotly.io.to_html(fig, include_plotlyjs='cdn'))

    df['assignment'] = df.apply(lambda row: map_assignment(row), axis=1)

    print(df.head())

    with open(Path(args.output)/f'{title}-blast-results.csv', 'w') as out:
        out.write(df.to_csv(index=False, header=True, lineterminator='\n'))


if __name__ == "__main__":
    sys.exit(main())

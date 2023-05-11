#!/usr/bin/env python3

import argparse
import itertools
import os
import pickle as p
import re
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from sklearn.manifold import TSNE


def get_args():
    parser = argparse.ArgumentParser(description="Count K-mers")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="*",
        default=".",
        help="input file (can be more than 1)",
    )
    parser.add_argument(
        "-o", "--output", type=str, nargs="*", default=".", help="output directory"
    )
    parser.add_argument(
        "-l", "--length", type=int, nargs="?", default=4, help="k-mer length"
    )
    parser.add_argument(
        "--load",
        type=str,
        nargs="?",
        default=False,
        help="filepath if loading from disk",
    )
    parser.add_argument("-p", "--plot", action="store_true", help="generate heat plot")
    parser.add_argument("-m", "--method",
                        choices=['contig', 'file'],
                        default='contig',
                        help='How file inputs are handled. Should each contig be a data point, or each file')
    return parser.parse_args()


def kmer_counter(files, method, kmer_length):
    """
    method="file" means arrange the dataframe with each filename as the index,
    method="contig" means arrange the dataframe with each contig header as the index

    "file" is best used for reads in fasta format, where which read a kmer came from isn't useful info really.
    "contig" is best for the assembled WGA fasta files, where we care about each contig as it's own data source.
    """

    nucleotides = ["A", "T", "C", "G"]

    a = list(itertools.product(nucleotides, repeat=kmer_length))
    df = pd.DataFrame(columns=["".join(x) for x in a])

    for file in files:
        name = file.stem
        print(name)
        sys.stdout.flush()

        with open(file, "r") as f:
            contigs = f.read().split(">")[1:]
        kmer_dict = {"".join(x): 0 for x in a}

        headers = [x.split("\n")[0] for x in contigs]
        seqs = ["".join(x.split("\n")[1:]).strip(" \n\t\r") for x in contigs]

        time_tracker = time.time()

        if method == "contig":
            for i, seq in enumerate(seqs):
                if i % 5000 == 0 and i != 0:
                    print(
                        f"{i} / {len(seqs)}\t-\t5000 seqs in {time.time()-time_tracker}"
                    )
                    sys.stdout.flush()
                    time_tracker = time.time()

                header = headers[i].split(" ")[0]

                for ii in range(0, len(seq), 1):
                    if ii + 4 > len(seq):
                        break
                    else:
                        s = seq[ii : ii + kmer_length]
                        if "N" not in s:
                            kmer = s
                            kmer_dict[kmer] += 1

                for k, val in kmer_dict.items():
                    kmer_dict[k] = val / (len(seq) // kmer_length)

                df.loc[header] = kmer_dict

        elif method == "file":
            kmers_seen_count = 0
            for i, seq in enumerate(seqs):
                if i % 1000000 == 0 and i != 0:
                    print(
                        f"\t{i} / {len(seqs)}\t-\t1,000,000 seqs in {time.time()-time_tracker:.2f}"
                    )
                    sys.stdout.flush()
                    time_tracker = time.time()

                for ii in range(0, len(seq), 1):
                    if ii + 4 > len(seq):
                        break
                    else:
                        s = seq[ii : ii + kmer_length]
                        if "N" not in s:
                            kmer = s
                            kmer_dict[kmer] += 1
                            kmers_seen_count += 1

            for k, val in kmer_dict.items():
                kmer_dict[k] = val / kmers_seen_count

            df.loc[name] = kmer_dict

        else:
            raise ValueError("Invalid method selected for kmer counting.")

    return df


def dataloader(path):
    with open(path, "rb") as infile:
        df = p.load(infile)

    return df


def testing_new_count_method(files):
    nucleotides = ["A", "T", "C", "G"]

    a = list(itertools.product(nucleotides, repeat=4))
    df = pd.DataFrame(columns=["".join(x) for x in a])
    files = list(files)

    for file in files:
        kmers_seen_count = 0
        t = time.time()
        name = file.stem
        print(name, end='\t')
        sys.stdout.flush()

        kmer_dict = {"".join(x): 0 for x in a}

        with open(file, "r") as f:
            for line in f:
                if "N" not in line:
                    kmer_dict[line.rstrip('\n')] += 1
                    kmers_seen_count += 1

        for k, val in kmer_dict.items():
            kmer_dict[k] = float(val / kmers_seen_count)

        print(name, time.time() - t)
        df.loc[name] = kmer_dict

    with open("read_kmer_counts.p", "wb") as o:
        p.dump(df, o)


def main():
    args = get_args()
    print(args.__dict__)
    sys.stdout.flush()

    if args.load:
        df = dataloader(args.load)

    else:
        files = [Path(x) for x in args.input]
        # df = testing_new_count_method(files)

        df = kmer_counter(files, args.method, args.length)

    print(df.head())
    sys.stdout.flush()

    # TODO: fix this. Output should be single string arg with default value of False
    Path(args.output[0]).mkdir(parents=True, exist_ok=True)
    os.chdir(Path(args.output[0]))

    print(os.getcwd())
    sys.stdout.flush()

    with open("contig_kmer_counts.p", "wb") as o:
        p.dump(df, o)

    if args.plot:
        df_log = np.log2(df)
        plot = px.imshow(df_log.T, x=list(df.index), title="Log scale")

        with open("../heatmap-plot.html", "w") as plot_file:
            plot_file.write(to_html(plot, include_plotlyjs="cdn"))


if __name__ == "__main__":
    sys.exit(main())




#!/usr/bin/env python3
import click
import pandas as pd


@click.command()
@click.option("--input", "-i", type=click.Path(exists=True))
def main(input):
    df = pd.read_csv(input, sep="\t", index_col=0)
    total_raw_counts = df.expected_count.values.sum()
    print(f"Total raw counts: {total_raw_counts}")

if __name__ == "__main__":
    main()

from ast import Num
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_tpm_for_sample(counts: np.array, effective_length: np.array) -> np.array:
    """
    Calculates the TPM for all genes in a dataframe.
    """
    if len(counts) != len(effective_length):
        raise ValueError(
            "Number of elements of counts and effective length must be the same."
            f"{len(counts)}: counst != {len(effective_length)}: effective_length"
        )
    transcript_per_kbp = counts / (effective_length / 1e3)
    tpm = transcript_per_kbp * 1e6 / np.sum(transcript_per_kbp)
    return tpm


def calculate_tpm(counts: pd.DataFrame, effective_length: pd.DataFrame) -> pd.DataFrame:
    counts = counts.copy()
    for column in counts.columns:
        counts[column] = calculate_tpm_for_sample(
            counts[column].values.astype(np.float32),
            effective_length.effective_length.values,
        )
    return counts


def read_effective_length(path: Path) -> pd.DataFrame:
    """
    Reads the effective length of all genes from a file.
    """
    return pd.read_csv(path, sep="\t", index_col=0)


def filter_by_tpm_and_count_passed(
    tpm_df: pd.DataFrame, threshold: Num
) -> dict:
    """
    count for each column is greater than threshold
    """
    filtered = tpm_df[tpm_df.apply(lambda x: x > threshold, axis=0)]
    # count non nan values of each column, return a dict, with column name as key
    count_dict = filtered.count().to_dict()
    return count_dict


def calculate_tpm_wrapper(
    counts: pd.DataFrame, effective_length_path: Path
) -> pd.DataFrame:
    """
    Calculates the TPM for all genes in a dataframe.
    """
    effective_length = read_effective_length(effective_length_path)
    tpm_df = calculate_tpm(counts, effective_length)
    return tpm_df

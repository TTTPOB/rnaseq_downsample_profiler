#!/usr/bin/env python3
from unittest import result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.downsample import *
from helpers.tpm_calculate import *
from multiprocessing import Pool

# %load_ext autoreload
# %autoreload 2

path = "/NAS392047/projects/epi_safety/20220111/SS2_mESC/matrix/counts.matrix"
effective_length_path = "/ds918_208/shared/genomics/mouse/GRCm38_2019/Annotations/GRCm38_gencode.m23/effective_length/gencode_vm23_effective_length.tsv"

raw_counts = pd.read_csv(path, sep="\t", index_col=0)

effective_length = read_effective_length(effective_length_path)
length_zero_gene = effective_length.query("effective_length == 0").index
# remove same gene from effective length
effective_length = effective_length.drop(length_zero_gene, axis=0)

# remove length zero genes from raw counts
raw_counts = raw_counts.drop(length_zero_gene, axis=0)
raw_counts = raw_counts[["SS2_WT_1", "SS2_WT_2", "SS2_WT_3"]]


def process_wrapper(
    counts: pd.DataFrame, rate: float, tpm_threshold: float, chunk: Num
) -> pd.DataFrame:
    filtered_counts = downsample(counts, rate, chunk)
    tpm = calculate_tpm(filtered_counts, effective_length)
    expressed_gene_count = filter_by_tpm_and_count_passed(tpm, tpm_threshold)
    return expressed_gene_count


def paramater_generator(rate_list: np.array):
    for rate in rate_list:
        yield raw_counts, rate, 1, 10000


rate_list = np.concatenate(
    (np.arange(0, 0.1, 0.01), np.arange(0.1, 1.0, 0.1), np.arange(0.9, 1.01, 0.01))
)
with Pool(24) as p:
    tpm_gt_2_list = p.starmap(process_wrapper, paramater_generator(rate_list))

expressed_gene_df = pd.DataFrame(tpm_gt_2_list, index=rate_list)

expressed_gene_df
plt.figure(2)
plt.plot(
    1 - rate_list,
    expressed_gene_df["SS2_WT_1"].values,
)
plt.xlabel("percent coverage")
plt.ylabel("number of expressed genes (TPM > 2)")
plt.savefig("plots/expressed_gene_vs_drop_out_rate.png")

encode_rate_list = np.logspace(0,1,10)/10

encode = pd.read_csv("./ENCFF910TAZ.tsv", sep="\t", index_col=0)
encode_eff_len = encode.effective_length.values
encode_counts = encode.expected_count.values
# get index of zero in encode_eff_len
encode_zero_index = encode_eff_len == 0
# remove these indexes from encode_eff_len and encode_counts
encode_eff_len = encode_eff_len[~encode_zero_index]
encode_counts = encode_counts[~encode_zero_index]


def encode_param_generator(rate_list: np.array):
    for rate in rate_list:
        yield encode_counts, encode_eff_len, rate, 2, 10000


def encode_command_wrapper(counts, effective_length, rate, threshold, chunk):
    dropped = random_subtract_from_array(counts, rate, chunk)
    tpm = calculate_tpm_for_sample(dropped, effective_length)
    filtered_counts = sum(tpm > threshold)
    return filtered_counts



with Pool(24) as p:
    filtered_count_result = p.starmap(
        encode_command_wrapper, encode_param_generator(encode_rate_list)
    )
filtered_count_result
encode_rate_list
#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
from utils.evaluation_tools import correlation_base_peak
import argparse


def correlation_base_functional_elements(
                    predicted_file,
                    experiment_file,
                    path_peak,
                    length=None,
                    window_size=128,
                    include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                    ):
    """The correlation between the predicted results and the experimental data is calculated in the area around the functional element"""
    correlation, spe = correlation_base_peak(
                                    predicted_file,
                                    experiment_file,
                                    path_peak,
                                    include_chr=include_chr,
                                    window_size=window_size,
                                    length=length,
                                    )
    
    return correlation, spe


def main():
    # Usage: python /local/zzx/code/BioSeq2Seq/src/HistoneModification/evaluation/correlation/base_FE/corr_base_tre.py --exper /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.exper.bigWig --pre /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.pred.bw --fe /local/zzx/code/BioSeq2Seq/genome_regions/FE_file/GM12878/promoter.bed --window 128
    parser = argparse.ArgumentParser(description="Calculate correlation near fuinctional elements")
    parser.add_argument("--exper", dest="bw_ground_truth", type=str, help="Path of the ground truth bigWig file")
    parser.add_argument("--pre", dest="bw_predicted", type=str, help="Path of the predicted bigWig file")
    parser.add_argument("--fe", dest="FE_file", type=str, help="Path of the Functional elements file")
    parser.add_argument("--window", dest="window_size", default=128, type=int, help="Window size")
    parser.add_argument("-w", dest="peak_width", default=1280, type=int, help="peak width")
    parser.add_argument("--chr", dest="chromosome", default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'], nargs='*', help="Chromosome for evaluation")
    args = parser.parse_args()

    include_chr = args.chromosome
    window_size = args.window_size
    bw_ground_truth = args.bw_ground_truth
    bw_predicted = args.bw_predicted
    path_peak = args.FE_file
    peak_width = args.peak_width

    correlation, spe = correlation_base_functional_elements(
                                    bw_predicted,
                                    bw_ground_truth,
                                    path_peak,
                                    include_chr=include_chr,
                                    window_size=window_size,
                                    length=peak_width,
                                    )

    print(correlation, spe)
                        


if __name__ == '__main__':
    main()

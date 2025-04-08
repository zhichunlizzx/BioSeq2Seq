#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import numpy as np
from sklearn.metrics import mean_squared_error
import pyBigWig
import os
import argparse

def mse(pre_data, exper_data):
    return mean_squared_error(exper_data, pre_data)

def mse1obs(y_true, y_pred):
    "the MSE at the top 1% of genomic positions ranked by experimental signal"
    n = int(y_true.shape[0] * 0.01)
    y_true_sorted = np.sort(y_true)
    y_true_top1 = y_true_sorted[-n]
    idx = y_true >= y_true_top1

    return mse(y_true[idx], y_pred[idx])


def mse1imp(y_true, y_pred):
    "the MSE at the top 1% of genomic positions ranked by predicted signal"
    n = int(y_true.shape[0] * 0.01)
    y_pred_sorted = np.sort(y_pred)
    y_pred_top1 = y_pred_sorted[-n]
    idx = y_pred >= y_pred_top1

    return mse(y_true[idx], y_pred[idx])


def mse_base_top(bw_pred, bw_exper, resolution=128, include_chr=[]):
    "read value from bigwig file and return MSE Imp and MSE Obs"
    bw_open_pred = pyBigWig.open(bw_pred)
    bw_open_exper = pyBigWig.open(bw_exper)

    chroms = bw_open_pred.chroms()

    bw_values_pred = np.asarray([])
    bw_values_exper = np.asarray([])
    
    # read values of every chromosome in include_chr
    for chr in include_chr:
        end = chroms[chr] // resolution * resolution
        bw_values_chr_pred = bw_open_pred.values(chr, 0, end, numpy=True)
        bw_values_chr_exper = bw_open_exper.values(chr, 0, end, numpy=True)

        # remove abnormal values
        bw_values_chr_pred[np.isnan(bw_values_chr_pred)] = 0
        bw_values_chr_pred[np.isinf(bw_values_chr_pred)] = 0

        bw_values_chr_exper[np.isnan(bw_values_chr_exper)] = 0
        bw_values_chr_exper[np.isinf(bw_values_chr_exper)] = 0

        bw_values_chr_pred = np.mean(bw_values_chr_pred.reshape(-1, resolution), axis=-1)
        bw_values_chr_exper = np.mean(bw_values_chr_exper.reshape(-1, resolution), axis=-1)

        bw_values_pred = np.append(bw_values_pred, bw_values_chr_pred)
        bw_values_exper = np.append(bw_values_exper, bw_values_chr_exper)

    mseObs = mse1obs(bw_values_exper, bw_values_pred)
    mseImp = mse1imp(bw_values_exper, bw_values_pred)
    
    return mseObs, mseImp


def mse_base_chr(bw_pred, bw_exper, resolution=128, include_chr=[]):
    "the MSE between predicted and expriment in whole chromosome or whole genome region"
    bw_open_pred = pyBigWig.open(bw_pred)
    bw_open_exper = pyBigWig.open(bw_exper)

    chroms = bw_open_pred.chroms()

    mseGlobal = 0
    n = 0
    # read values of every chromosome in include_chr
    for chr in include_chr:
        end = chroms[chr] // resolution * resolution
        bw_values_chr_pred = bw_open_pred.values(chr, 0, end, numpy=True)
        bw_values_chr_exper = bw_open_exper.values(chr, 0, end, numpy=True)

        # remove abnormal values
        bw_values_chr_pred[np.isnan(bw_values_chr_pred)] = 0
        bw_values_chr_pred[np.isinf(bw_values_chr_pred)] = 0

        bw_values_chr_exper[np.isnan(bw_values_chr_exper)] = 0
        bw_values_chr_exper[np.isinf(bw_values_chr_exper)] = 0

        bw_values_chr_pred = np.mean(bw_values_chr_pred.reshape(-1, resolution), axis=-1)
        bw_values_chr_exper = np.mean(bw_values_chr_exper.reshape(-1, resolution), axis=-1)

        mseGlobal += ((bw_values_chr_exper - bw_values_chr_pred)**2).sum()
        n += len(bw_values_chr_exper)
    
    return mseGlobal / n


def mse_base_genome_regions(bw_pred, bw_exper, resolution=128, include_chr=[], bedfile='', width=None):
    """
    compute the MSE based chromsome regions include Promoter, Enhancer, Gene.
    Args:
        bedfile: file of Promoter, Enhancer or Gene.
        width: specifies the length of each region

    """
    bw_open_pred = pyBigWig.open(bw_pred)
    bw_open_exper = pyBigWig.open(bw_exper)

    bed_list = np.loadtxt(bedfile, dtype='str')

    mseRegions = 0
    n = 0
    for bed_item in bed_list:
        chr = bed_item[0]
        if chr not in include_chr:
            continue
        
        start = int(bed_item[1]) // resolution * resolution
        end = int(bed_item[2]) // resolution * resolution

        # convert each region to specific width
        if width is not None:
            mid = (start + end) // 2
            start = int(mid - width // 2)
            end = int(mid + width / 2)

        if start == end:
            end = start + resolution
        
        bw_values_chr_pred = bw_open_pred.values(chr, start, end, numpy=True)
        bw_values_chr_exper = bw_open_exper.values(chr, start, end, numpy=True)

        bw_values_chr_pred[np.isnan(bw_values_chr_pred)] = 0
        bw_values_chr_pred[np.isinf(bw_values_chr_pred)] = 0

        bw_values_chr_exper[np.isnan(bw_values_chr_exper)] = 0
        bw_values_chr_exper[np.isinf(bw_values_chr_exper)] = 0

        bw_values_chr_pred = np.mean(bw_values_chr_pred.reshape(-1, resolution), axis=-1)
        bw_values_chr_exper = np.mean(bw_values_chr_exper.reshape(-1, resolution), axis=-1)

        mseRegions += ((bw_values_chr_exper - bw_values_chr_pred)**2).sum()
        n += len(bw_values_chr_exper)

    return mseRegions / n


def call_mse(
            bw_ground_truth,
            bw_predicted,
            resolution=128,
            include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
            fe_dir='',
            ):
    """ caculate six types of mse"""
    # mseGlobal
    mseGlobal = round(mse_base_chr(bw_predicted, bw_ground_truth, include_chr=include_chr, resolution=resolution), 5)

    # mseGene
    protein_coding_gene_file = os.path.join(fe_dir, 'genebody.bed')
    mseGene = round(mse_base_genome_regions(bw_predicted, bw_ground_truth, include_chr=include_chr, bedfile=protein_coding_gene_file, resolution=resolution), 2)

    # mseProm
    promoter_file = os.path.join(fe_dir, 'promoter.bed')
    mseProm = round(mse_base_genome_regions(bw_predicted, bw_ground_truth, include_chr=include_chr, bedfile=promoter_file, width=1280, resolution=resolution), 2)

    # mseEnh
    enhancer_file = os.path.join(fe_dir, 'enhancer.bed')
    mseEnh = round(mse_base_genome_regions(bw_predicted, bw_ground_truth, include_chr=include_chr, bedfile=enhancer_file, width=1280, resolution=resolution), 2)

    # mseObs, mseImp
    mseObs, mseImp = mse_base_top(bw_predicted, bw_ground_truth, include_chr=include_chr, resolution=resolution)
    mseObs, mseImp = round(mseObs, 2), round(mseImp, 2)

    return mseGlobal, mseGene, mseProm, mseEnh, mseObs, mseImp


def get_genome_wide_mse(bw_predicted,
                        bw_ground_truth,
                        fe_dir=None,
                        resolution=128,
                        include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                        'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                        'chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
                        ):
    mseGlobal, mseGene, mseProm, mseEnh, mseObs, mseImp = call_mse(bw_ground_truth,
                                                                   bw_predicted,
                                                                   fe_dir=fe_dir,
                                                                   resolution=resolution,
                                                                   include_chr=include_chr
                                                                   )

    print('mseGlobal: %s' % mseGlobal)
    print('mseGene: %s' % mseGene)
    print('mseProm: %s' % mseProm)
    print('mseEnh: %s' % mseEnh)
    print('mseObs: %s' % mseObs)
    print('mseImp: %s' % mseImp)

    return mseGlobal, mseGene, mseProm, mseEnh, mseObs, mseImp


if __name__ == '__main__':
    # python /local/zzx/code/BioSeq2Seq/src/HistoneModification/evaluation/MSE/mse_genome_wide.py --exper /local/zzx/code/BioSeq2Seq/test_samples/arcsinh/H3k4me1.bigWig --pre /local/zzx/code/BioSeq2Seq/test_samples/arcsinh/H3k4me1.bw --fe /local/zzx/code/BioSeq2Seq/genome_regions/FE_file/GM12878 
    parser = argparse.ArgumentParser(description="Calculate MSE for genome regions")
    parser.add_argument("--exper", dest="bw_ground_truth", type=str, help="Path to the ground truth bigWig file")
    parser.add_argument("--pre", dest="bw_predicted", type=str, help="Path to the predicted bigWig file")
    parser.add_argument("--fe", dest="FE_file_dir", type=str, help="Path to the FE file")
    parser.add_argument("--resolution", dest="resolution", default=128, type=int, help="Window size")
    parser.add_argument("--chr", dest="chromosome", default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'], nargs='*', help="Chromosome for evaluation")
    args = parser.parse_args()
    
    include_chr = args.chromosome
    resolution = args.resolution
    bw_ground_truth = args.bw_ground_truth
    bw_predicted = args.bw_predicted

    if args.FE_file_dir is None:
        fe_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../genome_regions/TRE_file/%s' % cellline))
    else:
        fe_dir = args.FE_file_dir


    mseGlobal, mseGene, mseProm, mseEnh, mseObs, mseImp = get_genome_wide_mse(
                                                                              bw_predicted,
                                                                              bw_ground_truth,
                                                                              fe_dir=fe_dir,
                                                                              resolution=resolution,
                                                                              include_chr=include_chr
                                                                              )

                
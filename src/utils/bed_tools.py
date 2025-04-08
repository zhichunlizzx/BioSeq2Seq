#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import numpy as np
import subprocess
import os



def write_chromsize(chrom_size, outpath):
    """write bed file, like chr 0, chromosome length"""
    # chromosome length file
    with open(outpath, 'w') as w_obj:
        for chr in chrom_size:
            w_obj.write(chr + '\t' + str(chrom_size[chr][0][1]) + '\n')

    return 1


def peak_bed_2_bigwig(whole_genome_size,
                    label_file,
                    item,
                    window_size=128,
                    outdir=os.path.dirname(os.path.abspath(__file__)),
                    include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                      'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                      'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
                    ):
    """covert bed file to 0, 1 signal"""
    reference_genome_idx = os.path.join(outdir, 'idx.fai')

    # whole genome regions
    regions = []
    for chr in whole_genome_size:
        if chr in include_chr:
            regions.append([chr, 0, whole_genome_size[chr][0][-1] // window_size * window_size])


    # sort label 
    sort_label = os.path.join(outdir, item+'.sort.bed')
    cmd_bedSort = 'sort-bed ' + label_file + ' > ' + sort_label
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()
    # merge label
    merge_label = os.path.join(outdir, item+'.merge.bed')
    cmd_merge = 'bedtools merge -i ' + sort_label + ' > ' + merge_label
    p = subprocess.Popen(cmd_merge, shell=True)
    p.wait()
    # sort label
    sort_label = os.path.join(outdir, item+'.sort.bed')
    cmd_bedSort = 'sort-bed ' + merge_label + ' > ' + sort_label
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()

    bedGraph_label = os.path.join(outdir, item+'.bedgraph')

    labels = np.loadtxt(sort_label, dtype='str', delimiter='\t')

    # (chr start, end) --> (chr, start, end, 1)
    with open(bedGraph_label, 'w') as w_obj:
        for label in labels:
            if label[0] in include_chr:
                w_obj.write('\t'.join(label)+'\t1\n')

    bw_label_file = os.path.join(outdir, item+'.bw')

    cmd = ['bedGraphToBigWig', bedGraph_label, reference_genome_idx, bw_label_file]
    subprocess.call(cmd)

    cmd_rm = ['rm', '-f', merge_label, sort_label, bedGraph_label]
    subprocess.call(cmd_rm)

    return 1


def read_peak_to_dict(path, include_chr):
    """read peak regions from file (chr, start, end)"""
    peaks = np.loadtxt(path, dtype='str')[:, :3]
    peak_dict = {}

    for chr in include_chr:
        peak_dict[chr] = []

    for peak in peaks:
        if peak[0] in include_chr:
            peak_dict[peak[0]].append({'start': peak[1], 'end': peak[2]})

    return peak_dict


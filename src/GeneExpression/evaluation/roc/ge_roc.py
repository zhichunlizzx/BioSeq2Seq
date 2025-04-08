#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import os
import pyBigWig
import sys
sys.path.insert(0, sys.path[0]+"/../")
sys.path.append('/local/zzx/code/TRE/BioSeq2Seq')
import numpy as np
import pyBigWig
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from utils.bed_tools import peak_bed_2_bigwig
from model_function.functions import bw_2_chromosome_size
import argparse


def read_values(predicted_file, label_file, regions, window_size):
    """
    read values from predicted bigwig file and label file
    Args:
        predicted_file: predicted histone modification file.
        label_file (bw): label file (0 or 1).
        regions: chromosome regions
        window_size: bin size used in evaluation
        width: specifies the width of peak
    """
    predicted_open = pyBigWig.open(predicted_file, 'r')
    label_open = pyBigWig.open(label_file, 'r')

    max = predicted_open.header()['maxVal']

    predicted_data = np.asarray([])
    label_data = np.asarray([])
    for region in regions:
        pre = predicted_open.values(region[0], int(region[1]), int(region[2]), numpy=True).astype('float16').reshape((-1, window_size))
        label = label_open.values(region[0], int(region[1]), int(region[2]), numpy=True).astype('float16').reshape((-1, window_size))
        label[np.isnan(label)] = 0
        label[np.isinf(label)] = 0

        pre[np.isnan(pre)] = 0
        pre[np.isinf(pre)] = 0
        
        pre = np.mean(pre, axis=-1) / max
        pre[pre>1] = 1

        label = np.max(label, axis=-1).astype('int')

        predicted_data = np.append(predicted_data, pre)
        label_data = np.append(label_data, label)

    predicted_open.close()
    label_open.close()

    return predicted_data, label_data

def read_predicted_experiment_values(bw_file, label_file, outdir, item, include_chr=['chr22'], window_size=128):
    """
    get predicted values and ground truth based specified regions.
    Args:
        bw_file: predicted bigwig file.
        label_file: peak file of experiment (bed).
        outdir: out dir.
        item: histone modification type.
        include_chr: selected chromosomes.
        window_size: bin size.
    """
    whole_genome_size = bw_2_chromosome_size(bw_file=bw_file, outdir=outdir)

    bed_2_bw = peak_bed_2_bigwig(whole_genome_size,
                                 label_file,
                                 item,
                                 window_size=window_size,
                                 outdir=outdir,
                                 include_chr=include_chr
                                 )
    
    # whole genome regions
    regions = []
    for chr in whole_genome_size:
        if chr in include_chr:
            regions.append([chr, 0, whole_genome_size[chr][0][-1] // window_size * window_size])


    bw_label_file = os.path.join(outdir, item+'.bw')
    predicted_values, label_values = read_values(bw_file, bw_label_file, regions, window_size)

    return predicted_values, label_values
 

def draw_roc_prc(predicted_file,
                 label_file,
                 outdir,
                 include_chr=['chr22'],
                 color='darkorange',
                 window_size=128,
                 ):
    "draw roc curves and pr curves"
    item = os.path.basename(predicted_file)

    roc_out = os.path.join(outdir, '%s.roc.pdf'%item)
    prc_out = os.path.join(outdir, '%s.prc.pdf'%item)
    if not os.path.exists(outdir):
        os.makedirs(outdir) 

    
    predicted_values, label_values = read_predicted_experiment_values(predicted_file, label_file, outdir, item, include_chr=include_chr, window_size=window_size)
    
    # roc
    fpr, tpr, thresholds = roc_curve(label_values, predicted_values, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # prc
    precision, recall, _ = precision_recall_curve(label_values, predicted_values)
    prc = average_precision_score(label_values, predicted_values, average='macro', sample_weight=None)

    # draw the precision-recall curve
    plt.plot(fpr, tpr, lw=2, color=color, label=f'{item}(AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel('False Positive Rate', fontdict={'size': 15})
    plt.ylabel('True Positive Rate', fontdict={'size': 15})
    plt.title('Receiver Operating Characteristic', fontdict={'size': 15})
    plt.legend(loc="lower right", prop={'size': 10})
    plt.savefig(roc_out)
    plt.clf()

    # draw the receiver operator characteristic curve
    plt.plot(recall, precision, lw=2, label=f'{item}(PRC = {prc:.4f})')
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel("Recall", fontdict={'size': 15})
    plt.ylabel("Precision", fontdict={'size': 15})
    plt.title('Precision-Recall Curve', fontdict={'size': 15})
    plt.legend(loc="lower left", prop={'size': 10})
    plt.savefig(prc_out)
    plt.clf()

    return roc_auc, prc
 
if __name__ == '__main__':
    # Useage: python /local/zzx/code/BioSeq2Seq/src/FE/evaluation/roc/fe_roc.py --exper /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.bed --pre /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.pred.bw -o /local/zzx/code/BioSeq2Seq/test_samples
    parser = argparse.ArgumentParser(description="Calculate AUROC and AUPR for genome regions")
    parser.add_argument("--exper", dest="bed_ground_truth", type=str, help="Path to the ground truth bed file")
    parser.add_argument("--pre", dest="bw_predicted", type=str, help="Path to the predicted bigWig file")
    parser.add_argument("-o", dest="outdir", type=str, help="Output directory")
    parser.add_argument("--resolution", dest="resolution", default=128, type=int, help="Window size")
    parser.add_argument("--chr", dest="chromosome", default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'], nargs='*', help="Chromosome for evaluation")
    args = parser.parse_args()

    
    include_chr = args.chromosome

    outdir = args.outdir
    label_file = args.bed_ground_truth
    predicted_file = args.bw_predicted
    
    roc_auc, prc = draw_roc_prc(predicted_file, label_file, outdir, window_size=args.resolution, include_chr=include_chr)
    print(roc_auc)
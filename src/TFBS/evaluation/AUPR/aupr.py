import os
import pyBigWig
import sys
sys.path.insert(0, sys.path[0]+"/../")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
import numpy as np
from model_function.functions import bw_2_chromosome_size
import pyBigWig
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils.bed_tools import peak_bed_2_bigwig


def read_values(bw_file, label_file, regions, window_size, max=192, min=0.5, include_chr=['chr22']):
    predicted_open = pyBigWig.open(bw_file, 'r')
    label_open = pyBigWig.open(label_file, 'r')

    max = predicted_open.header()['maxVal']
    chroms = predicted_open.chroms()

    predicted_data = np.asarray([])
    label_data = np.asarray([])

    for chr in chroms:
        if chr not in include_chr:
            continue
        chr_end = int(chroms[chr])
        num_bin = chr_end // window_size
        pre = np.nan_to_num(np.array(predicted_open.stats(chr, 0, chr_end, type='max', nBins=num_bin, exact=True), dtype=float))
        label = np.nan_to_num(np.array(label_open.stats(chr, 0, chr_end, type='max', nBins=num_bin, exact=True), dtype=float))

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
 

def call_aupr(tf_pred, tf_label, outpath, include_chr=['chr22'], window_size=1000, recall_threshold=0.05):
    tf = os.path.splitext(os.path.basename(tf_pred))[0]
    print(tf)

    predicted_values, label_values = read_predicted_experiment_values(tf_pred, tf_label, outpath, tf, include_chr=include_chr, window_size=window_size)

    precision, recall, _ = precision_recall_curve(label_values, predicted_values)
    aupr = average_precision_score(label_values, predicted_values, average='macro', sample_weight=None)

    index = next(i for i, v in enumerate(recall) if v < recall_threshold)
    index = index - 1
    precision_recall_threshold = precision[index]

    print('%s: AUPR: %f, Precision at %d%% Recall: %f' % (tf, aupr, int(recall_threshold*100), precision_recall_threshold))

    return aupr
    

if __name__ == '__main__':
    # Useage:
    include_chr = ['chr22']
    window_size = 200
    outpath = '/local/zzx/code/BioSeq2Seq/test_samples/TF/out'
    tf_pred = '/local/zzx/code/BioSeq2Seq/test_samples/TF/CTCF.bw'
    tf_label = '/local/zzx/code/BioSeq2Seq/test_samples/TF/CTCF.label.bed'

    call_aupr(tf_pred, tf_label, outpath, include_chr=include_chr, window_size=window_size)

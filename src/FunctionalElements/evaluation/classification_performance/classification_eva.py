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
sys.path.insert(0, sys.path[0]+"/../")
sys.path.append(os.path.abspath('.')+"/..")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from re import I
import numpy as np
from model_function.get_feature import get_target_feature
import subprocess
from tqdm import tqdm
from operator import itemgetter
import pyBigWig
from scipy.stats import multivariate_normal, pearsonr, norm
import math
from sample.wholegenome_samples import get_predicted_samples
from scipy.signal import find_peaks
import time
import statsmodels.api as sm


FRAC = 0.1

def convert_resolution(data, resolution):
    data = data[: (len(data) // resolution) * resolution]
    data = np.mean(data.reshape(-1, resolution), axis=-1)
    return data


def smooth_bw(bw_file, outdir, reference_genome_idx, chunk_size=12800, resolution=128):
    lowess_sm = sm.nonparametric.lowess

    target = 'smooth_genebody'
    bw_open = pyBigWig.open(bw_file, 'r')
    chroms_dict = bw_open.chroms()
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    bedGraphFile = os.path.join(outdir, '%s.bedGraph' % target)

    num_point_each_chunk = chunk_size // resolution
    with open(bedGraphFile, 'w') as w_obj:
        for chr in chroms_dict:
            # print(chr)
            chr_length = chroms_dict[chr]

            bw_values = bw_open.values(chr, 0, chr_length, numpy=True)
            
            for i in range(chr_length // chunk_size):
                region_values = bw_values[chunk_size*i:chunk_size*(i+1)]
                region_values = convert_resolution(region_values, resolution)
                x = np.arange(i*num_point_each_chunk, (i+1)*num_point_each_chunk)
                sm_y = lowess_sm(region_values, x, frac=FRAC, it=3, delta=1, return_sorted=False)

                start_list = x * resolution
                end_list = (x+1) * resolution
                
                for i in range(len(start_list)):
                    w_obj.write(chr + '\t' + str(start_list[i]) + '\t' + str(end_list[i]) + '\t' + str(sm_y[i]) + '\n')

    bw_open.close()

    bedgraph_path_sorted = os.path.join(outdir, '%s.sorted.bedGraph' % target)
    cmd_bedSort = 'sort-bed ' + bedGraphFile + ' > ' + bedgraph_path_sorted
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()
    bw_path = os.path.join(outdir, '%s.bw' % target)

    cmd = ['bedGraphToBigWig', bedgraph_path_sorted, reference_genome_idx, bw_path]
    subprocess.call(cmd)

    cmd_rm = ['rm', '-f', bedGraphFile]
    subprocess.call(cmd_rm)

    cmd_rm = ['rm', '-f', bedgraph_path_sorted]
    subprocess.call(cmd_rm)


def slice_peak(out_dir, smoothed_bw_file, resolution=128):
    peak_file = out_dir + '/genebody.bed'
    out_file = out_dir + '/further_slice_genebody.bed'

    peaks = np.loadtxt(peak_file, dtype='str')
    
    bw_open = pyBigWig.open(smoothed_bw_file, 'r')

    with open(out_file, 'w') as w_obj:
        for peak in peaks:
            # if sample[0] != 'chr22':
            #     continue
            start, end = int(peak[1]), int(peak[2])
            x = np.arange(start / resolution, end / resolution)
            # print(bw_open.header())
            peak_reads = bw_open.values(peak[0], start, end, numpy=True).astype('float16')

            peak_reads_minus = np.mean((peak_reads * -1).reshape(-1, resolution), axis=-1)

            peak_id, peak_property = find_peaks(peak_reads_minus, width=len(x)*0.02, prominence=0.8)
            peak_freq = x[peak_id]

            if len(peak_freq) > 0:
                w_obj.write(peak[0] + '\t' + peak[1] + '\t' + str(int(peak_freq[0] * resolution)) + '\n')
                for i in range(len(peak_freq))[1:]:
                    w_obj.write(peak[0] + '\t' + str(int((peak_freq[i-1] + 1) * resolution)) + '\t' + str(int(peak_freq[i] * resolution)) + '\n')
                
                w_obj.write(peak[0] + '\t' + str(int((peak_freq[-1] + 1) * resolution)) + '\t' + peak[2] +  '\n')
                
            else:
                w_obj.write(peak[0] + '\t' + peak[1] + '\t' + peak[2] + '\n')


def read_data(data_file, gap=2):
    print('start read file...')
    items = np.loadtxt(data_file, dtype='str', delimiter='\t')
    print('complete read file...')
    chr_list = np.unique(items[:, 0])
    # print(items[0])
    reads_list = np.asarray([])
    for chr in chr_list:
        # print('%s...'%chr)
        chr_items = np.where(items[:, 0]==chr)[0]
        start, end = chr_items[0], chr_items[-1]
        chose_list = np.arange(start, end, gap)

        chr_reads = np.asarray(items[chose_list, -1])
        reads_list = np.append(reads_list, chr_reads)
    reads_list = np.asarray(reads_list, dtype='float')
    reads_list = reads_list[reads_list > 0]

    return np.log(reads_list)


def cov(read_depths):
    x0 = read_depths[:-4]
    x1 = read_depths[1:-3]
    x2 = read_depths[2:-2]
    x3 = read_depths[3:-1]
    x4 = read_depths[4:]

    x = np.asarray([x0, x1, x2, x3, x4])
    x = x.T

    cov_mat = np.cov(x, rowvar=False)
    return cov_mat


def mean_and_cov(read_depth_promoter_file):
    reads_list = read_data(read_depth_promoter_file)
    mean = np.mean(reads_list)
    std = np.std(reads_list)
    cov_mat = cov(read_depths=reads_list)
    return mean, std, cov_mat


def get_read_depth(bw_file, read_depth_bed, resolution=128):
    """whole genome read depth based resolution"""
    bw_open = pyBigWig.open(bw_file, 'r')
    chrom_length = bw_open.chroms()
    read_values = []
    flat_regions = []
    # read bigwig signal based resolution
    for chr, length in chrom_length.items():
        # print(chr, length)
        region_list = list(range(length // resolution))
        
        current_region = 0
        item_value = np.mean(bw_open.values(chr, 0*resolution, (0+1)*resolution))
        current_value = item_value
        read_values.append([chr, 0*resolution, (0+1)*resolution, item_value])
        flat_num = 0
        
        for region in region_list[1:]:
            item_value = np.mean(bw_open.values(chr, region*resolution, (region+1)*resolution))
            read_values.append([chr, region*resolution, (region+1)*resolution, item_value])

            if current_value == 0:
                continue
            if (abs((item_value / current_value) - 1)) < 0.1:
                flat_num += 1
            else:
                if flat_num >= 100:
                    flat_regions.append([chr, str(current_region*resolution), str((region-1)*resolution)])
                
                current_region = region
                current_value = item_value
                flat_num = 0

    # write to file
    with open(read_depth_bed, 'w') as w_obj:
        for r_value in read_values:
            w_obj.write(r_value[0] + "\t" + str(r_value[1]) + "\t" + str(r_value[2]) + "\t" + str(r_value[3]) + '\n')
    
    with open(os.path.join(os.path.dirname(read_depth_bed), 'meaningless_region.bed'), 'w') as w_obj:
        for region in flat_regions:
            w_obj.write('\t'.join(region) + '\n')

    return read_values


def rm_meaningless_region(read_depth_bed):
    """remove meaningless regions"""
    gap_file = os.path.join(os.path.dirname(read_depth_bed), 'meaningless_region.bed')
    cmd_subtract = 'bedtools subtract -a ' + read_depth_bed + ' -b ' + gap_file + ' > ' + 'temp ' + '&& mv temp ' + read_depth_bed
    p = subprocess.Popen(cmd_subtract, shell=True)
    p.wait()


def classified_eva(tn, fn, fp, tp):
    acc = (tp + tn) / (tp + fp + tn + fn)
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = (2 * pre * recall) / (pre + recall)
    return acc, pre, recall, f_score


def read_peaks(file, include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
              'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']):
    with open(file, 'r') as r_obj:
        peaks = r_obj.readlines()
        peaks = [peak[:-1].split('\t') for peak in peaks]
        peaks = [peak for peak in peaks if peak[0] in include_chr]
    
    return peaks


def write_peaks(file, include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
              'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']):
    peaks = read_peaks(file, include_chr)
    with open(file, 'w') as w_obj:
        for peak in peaks:
            w_obj.write("\t".join(peak) + "\n")
    return True


def peak_IOU(peak, extend=0):
    label_start, label_end = int(peak[1]), int(peak[2])

    len_label_peak = label_end - label_start

    intersect = int(peak[-1])
    # 修正因overlap时因extend导致的负数
    if intersect < 0:
        intersect += extend
        len_label_peak += 2 * extend
    
    iou = intersect / len_label_peak

    return iou


def bw_2_chromosome_size(bw_file, outpath, include_chr):
    """Read chromosome size from .bw file"""
    bw_open = pyBigWig.open(bw_file)
    chromsize = bw_open.chroms()

    with open(os.path.join(outpath, 'genome.idx'), 'w') as w_obj:
        for chr in chromsize:
            chromsize[chr] = [(0, chromsize[chr])]
            w_obj.write(chr + '\t' + '0' + '\t' + str(chromsize[chr][0][1]) + '\n')

    with open(os.path.join(outpath, 'genome.length'), 'w') as w_obj:
        for chr in chromsize:
            w_obj.write(chr + '\t' + str(chromsize[chr][0][1]) + '\n')
    
    return chromsize


def get_sample(peak, bw_file, resolution=128):
    x = []
    bw_object = pyBigWig.open(bw_file)
    start, end = int(peak[1]), int(peak[2])
    start = (start // resolution) * resolution
    end = (end // resolution) * resolution

    try:
        peak_read = np.asarray(bw_object.values(peak[0], start-2*resolution,  end+2*resolution))
    except:
        return 0.1

    inf_idx = np.isinf(peak_read)
    peak_read[inf_idx] = 0.00000001

    nan_idx = np.isnan(peak_read)
    peak_read[nan_idx] = 0.00000001

    zero_idx = (peak_read == 0)
    peak_read[zero_idx] = 0.00000001

    peak_read = np.mean(peak_read.reshape(-1, resolution), axis=-1)
    l_reads = len(peak_read)
    a = peak_read
    max_site = np.argmax(peak_read[2:-2])+2
    
    # 扩展两个4个元素后，如果长度小于9(原来小于5)
    if l_reads < 5+4:
        if max_site == 2:
            peak_read = peak_read[:-2]

        elif max_site == l_reads -2:
            peak_read = peak_read[2:]

        else:
            peak_read=peak_read[max_site-2: max_site+3]
    else:
        if max_site == 2:
            peak_read = peak_read[:-2]
        elif max_site == l_reads -2:
            peak_read = peak_read[2:]
        else:
            peak_read = peak_read[2:-2]
    try:
        max_site = np.argmax(peak_read)
    except:
        print(peak[0], start,  end)
        print(l_reads)
        print(a)
        print(peak_read)
        print(max_site)
        raise Exception('aa')

    max_site = np.argmax(peak_read)
    try:
        left_min_site = np.argmin(peak_read[:max_site])
    except:
        left_min_site = max_site
    try:
        right_min_site = np.argmin(peak_read[max_site:])
    except:
        right_min_site = max_site

    left_mid_site = math.ceil((left_min_site+max_site)/2)
    right_mid_site = math.ceil((right_min_site+max_site)/2)

    x = [
        np.log(peak_read[left_min_site]),
        np.log(peak_read[left_mid_site]),
        np.log(peak_read[max_site]),
        np.log(peak_read[right_mid_site]),
        np.log(peak_read[right_min_site]),
        ]
    
    return np.asarray(x)

def corvar(x1, x2):
    spear_corr = pearsonr(x1, x2)[0]
    std_x1 = np.std(x1)
    std_x2 = np.std(x2)

    return std_x1 * std_x2 * spear_corr

def cormat(peak_file, bw_file):
    # 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% 
    x0, x1, x2, x3, x4 = [], [], [], [], []

    with open(peak_file, 'r') as r_obj:
        lines = r_obj.readlines()

    peaks = [line[:-1].split('\t') for line in lines]
    
    bw_object = pyBigWig.open(bw_file)
    for peak in peaks:
        try:
            if int(peak[2]) - int(peak[1]) > 5000:
                peak_read = np.asarray(bw_object.values(peak[0], int(peak[1]),  int(peak[2])))
        except:
            print('Missing information about this region: ')
            print(peak[0], int(peak[1]),  int(peak[2]))
        

        inf_idx = np.isinf(peak_read)
        peak_read[inf_idx] = 0.00000001

        nan_idx = np.isnan(peak_read)
        peak_read[nan_idx] = 0.00000001

        zero_idx = (peak_read == 0)
        peak_read[zero_idx] = 0.00000001

        peak_read = np.sort(peak_read)
        x0.append(math.log(peak_read[0]))
        x1.append(math.log(peak_read[int(len(peak_read)*0.25)]))
        x2.append(math.log(peak_read[int(len(peak_read)*0.5)]))
        x3.append(math.log(peak_read[int(len(peak_read)*0.75)]))
        x4.append(math.log(peak_read[-1]))

    x = np.asarray([x0, x1, x2, x3, x4])
    x = x.T
    # # print(x)
    corrvar_matrix = np.cov(x, rowvar=False)
    mean = []
    for i in range(5):
        mean.append(np.mean(locals()['x%d' % i]))

    return corrvar_matrix, np.asarray(mean)
    

def write_regression_result(results,
                            out_path,
                            target_list,
                            reference_genome_idx,
                            seq_length=114688,
                            window_size=128,):
    """ 
    Write result to bigwig file

    Args:
        results: predicted result, {chr:[{start:xx, end:xx, result:xx}]}
        out_path: output path
        target_list: target sequencing data list
        reference_genome_idx: reference genome idx

    Return:
        None
    """
    seq_length = seq_length
    target_length = seq_length // window_size

    for j in range(len(target_list)):
            if os.path.isfile(os.path.join(out_path, target_list[j] + '.bedgraph')):
                os.remove(os.path.join(out_path, target_list[j] + '.bedgraph'))

    for chr in results:
        chr_result = results[chr]
        chr_result = sorted(chr_result, key=itemgetter('start'))
        for j in range(len(target_list)):
            with open(os.path.join(out_path, target_list[j] + '.bedgraph'), 'a') as w_obj:

                last_end = 0
                for item in chr_result:
                    if item['start'] >= last_end: 
                        for i in range(target_length):
                            if item['predicted'][i][j] != 0:
                                start = item['start'] + i * window_size
                                end = start + window_size
                                w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
        
                    else:
                        print(item)
                        gap_h = last_end - item['start']
                        h_start = gap_h // window_size
                        if item['predicted'][h_start][j] != 0:
                            w_obj.write(chr + '\t' + str(last_end) + '\t' + str(item['start'] + window_size * (h_start+1)) + '\t' + str(item['predicted'][h_start][j]) + '\n')
                        for i in range(h_start+1, target_length):
                            if item['predicted'][i][j] != 0:
                                start = item['start'] + i * window_size
                                end = start + window_size 
                                w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                last_end = item['end']

    for j in range(len(target_list)):
        bed_path = os.path.join(out_path, target_list[j] + '.bedgraph')
        bedgraph_path_sorted = os.path.join(out_path, target_list[j] + '_sorted.bedgraph')
        cmd_bedSort = 'sort-bed ' + bed_path + ' > ' + bedgraph_path_sorted
        p = subprocess.Popen(cmd_bedSort, shell=True)
        p.wait()

        bw_path = os.path.join(out_path, target_list[j] + '.bigwig')

        cmd = ['bedGraphToBigWig', bedgraph_path_sorted, reference_genome_idx, bw_path]
        subprocess.call(cmd)

        cmd_rm = ['rm', '-f', bed_path]
        subprocess.call(cmd_rm)

        cmd_rm = ['rm', '-f', bedgraph_path_sorted]
        subprocess.call(cmd_rm)

    return True


def bw_to_01_bw(
            samples,
            target_seq_file,
            target_list,
            chrom_size,
            out_path,
            threshold,
            seq_length=114688,
            window_size=128,
            ):
    """ 
    Write result to bigwig file

    Args:
        model: trained model
        samples: samples with length of 114688 bp, [num_of_samples, 3]
        reference_genome_file: reference genome file
        sequence_data_path: file path of sequcing data
        out_path: output path
        target_list: target sequencing data list

    Return:
        None
    """
    results = {}
    chr_length = {}
    for chr in np.unique(samples[:, 0]):
        results[chr] = []
        chr_length[chr] = chrom_size[chr][0][1]

    if not os.path.isdir(out_path):
            os.mkdir(out_path)

    reference_genome_idx = os.path.join(out_path, 'idx.fai')
    with open(reference_genome_idx, 'w') as w_obj:
        for chr in chrom_size:
            w_obj.write(chr + '\t' + str(chrom_size[chr][0][1]) + '\n')

    for i, sample in tqdm(enumerate(samples)):
        result = {}
        result['chr'] = sample[0]
        result['start'] = int(sample[1])
        result['end'] = int(sample[2])

        pro = get_target_feature([sample], [target_seq_file[0]], nan=0, window_width=window_size)[0]
        poly = get_target_feature([sample], [target_seq_file[1]], nan=0, window_width=window_size)[0]
        insu = get_target_feature([sample], [target_seq_file[2]], nan=0, window_width=window_size)[0]
        gene = get_target_feature([sample], [target_seq_file[3]], nan=0, window_width=window_size)[0]

        pred = np.concatenate([pro, poly, insu, gene], axis=-1)
        result['predicted'] = (pred >= threshold).astype('int')

        
        results[result['chr']].append(result)

    write_down = write_regression_result(results, out_path, target_list, reference_genome_idx, seq_length, window_size)
    
    return True


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re


def Bonferroni(path, outpath, target):
    bonferroni_outpath = os.path.join(outpath, 'bonferroni_%s_p_value.bed'%target)
    bonferroni_log_outpath = os.path.join(outpath, 'bonferroni_%s_log_p_value.bed'%target)

    with open(path, 'r') as r_obj:
        items = r_obj.readlines()
    items = np.asarray([item[:-1].split('\t') for item in items])

    p_value = items[:, -1].astype('float')

    num_peak = len(items)

    bonferroni_p_value = p_value * num_peak

    with open(bonferroni_outpath, 'w') as w_obj:
        for i in range(num_peak):
            w_obj.write(items[i][0] + '\t' + items[i][1] + '\t' + items[i][2] + '\t' + str(bonferroni_p_value[i])  + '\n')

    temp_sort_file = os.path.join(outpath, 'temp_sort_%s.bed'%target)
    cmd_bedSort = 'sort-bed ' + bonferroni_outpath + ' > ' + temp_sort_file
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()

    cmd_mv = ['mv', temp_sort_file, bonferroni_outpath]
    subprocess.call(cmd_mv)

    #######################
    # log p-value
    ######################
    bonferroni_p_value = np.log10(bonferroni_p_value) * -1
    bonferroni_p_value[bonferroni_p_value<0] = 0.01
    # bonferroni_p_value = np.tanh(np.square(bonferroni_p_value) / 50)
    

    with open(bonferroni_log_outpath, 'w') as w_obj:
        for i in range(num_peak):
            w_obj.write(items[i][0] + '\t' + items[i][1] + '\t' + items[i][2] + '\t' + str(bonferroni_p_value[i])  + '\n')

    temp_sort_file = os.path.join(outpath, 'temp_sort_%s.bed'%target)
    cmd_bedSort = 'sort-bed ' + bonferroni_log_outpath + ' > ' + temp_sort_file
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()

    cmd_mv = ['mv', temp_sort_file, bonferroni_log_outpath]
    subprocess.call(cmd_mv)

    reference_genome_idx = os.path.join(outpath, 'idx.fai')
    bonferroni_log_bw_path = os.path.join(outpath, 'bonferroni_%s_log_p_value.bw'%target)
    cmd = ['bedGraphToBigWig', bonferroni_log_outpath, reference_genome_idx, bonferroni_log_bw_path]
    subprocess.call(cmd)

    cmd_rm = ['rm', '-f', bonferroni_log_outpath]
    subprocess.call(cmd_rm)

    return True


def BenjaminiFDR(path, outpath, target):
    fdr_outpath = os.path.join(outpath, 'fdr_%s_p_value.bed'%target)

    with open(path, 'r') as r_obj:
        items = r_obj.readlines()
    items = np.asarray([item[:-1].split('\t') for item in items])

    p_value = items[:, -1].astype('float')

    num_peak = len(items)

    items = items[np.argsort(p_value)]
    
    with open(fdr_outpath, 'w') as w_obj:
        for i in range(num_peak):
            w_obj.write(items[i][0] + '\t' + items[i][1] + '\t' + items[i][2] + '\t' + str(float(items[i][3]) * (num_peak / (i + 1)))  + '\n')

    temp_sort_file = os.path.join(outpath, 'temp_sort_%s.bed'%target)
    cmd_bedSort = 'sort-bed ' + fdr_outpath + ' > ' + temp_sort_file
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()

    cmd_mv = ['mv', temp_sort_file, fdr_outpath]
    subprocess.call(cmd_mv)

    return True

def bw_2_peak_bw(bw_file, out_bw, whole_genome_size, threshold):
    #废案，慢且问题多
    bw_open_write = pyBigWig.open(out_bw, "w")
    bw_open_read = pyBigWig.open(bw_file, 'r')
    include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
              'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    
    for chr in include_chr:
        region = whole_genome_size[chr]
        start, end = region[0]
        bw_open_write.addHeader([(chr, end)], maxZooms=0)
        bw_read_data = bw_open_read.values(chr, start, end, numpy=True).astype('float16')
        bw_read_data = (bw_read_data >= threshold).astype('int').astype('float32')

        x = np.arange(start, end)

        chr_list = np.array([chr] * (end- start))

        bw_open_write.addEntries(chr_list,
                            x,
                            ends=x+1,
                            values=bw_read_data)
    bw_open_write.close()
    bw_open_read.close()


def bw_2_peak_bed(bw_list, outpath, whole_genome_size, threshold, target_list, seq_length=114688, window_size=128, include_chr=['chr22']):
    regions = get_predicted_samples(whole_genome_size,
                                    include_chr,
                                    seq_length,
                                    )
    # print(regions[:3])
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    # 用的四个TRE预测的测序深度中噪声奇异值最大的Genebody中的奇异值threshold，这样可以很好的囊括这些噪声
    bw_to_01_bw(regions,
                bw_list,
                target_list,
                whole_genome_size,
                outpath,
                threshold=threshold,
                seq_length=seq_length,
                window_size=window_size
                )

    for item in target_list:
        bw_file = os.path.join(outpath, item+'.bigwig')
        bed_file = os.path.join(outpath, item+'.bed')

        cmd = ['bigWigToBedGraph', bw_file, bed_file]
        subprocess.call(cmd)

        cmd_rm = ['rm', '-f', bw_file]
        subprocess.call(cmd_rm)

    with open(bed_file, 'r') as r_obj:
        items = r_obj.readlines()

    items = [item[:-1].split('\t') for item in items]
    
    with open(bed_file, 'w') as w_obj:
        for item in items:
            w_obj.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\n')


def peak_add_p_value(target_list, bw_list, outpath, corrvar_matrix=None, mean=None, bonferroni_direct=False, filter_threshold=1e-1):
    
    for i in range(len(target_list)):
        # 选择TRE
        target = target_list[i]
        print('start %s p-value evaluation' % target)
        target_bw = bw_list[i]
        bed_file = os.path.join(outpath, target+'.bed')

        # genebody使用波谷分割法分割后的region文件
        if target == 'genebody':
            bed_file = os.path.join(outpath, 'further_slice_genebody.bed')

        # negative
        # 初筛negative，最终的negative是使用p-value过滤后的positive的补集
        complement_file = os.path.join(outpath, '%s_complement.bed'%target)

        idx_file = os.path.join(outpath, 'genome.idx')
        cmd_subtract = 'bedtools subtract -a ' + idx_file + ' -b ' + bed_file + ' > ' + complement_file
        p = subprocess.Popen(cmd_subtract, shell=True)
        p.wait()

        p_value_bed_file = os.path.join(outpath, '%s_p_value.bed'%target)

        if not bonferroni_direct:
            if corrvar_matrix is not None:
                pass
            else:
                corrvar_matrix, mean = cormat(complement_file, target_bw)
                print('corrvar_matrix:', corrvar_matrix)
                print('mean:', mean)

            with open(bed_file, 'r') as r_obj:
                items = r_obj.readlines()
            items = [item[:-1].split('\t') for item in items]

            print('write %s p-value to bed file' % target)
            with open(p_value_bed_file, 'w') as w_obj:
                a = 0
                for item in items:

                    x = get_sample([item[0], int(item[1]), int(item[2])], target_bw)
                    x = 2 * mean - x
                    # multi_t cdf(x, loc=None, shape=1, df=1, allow_singular=False, *, maxpts=None, lower_limit=None, random_state=None)
                    area = multivariate_normal.cdf(x, mean=mean, cov=corrvar_matrix)

                    # if area > 0.5:
                    #     area = 1 - area
                    w_obj.write(item[0] + '\t' + item[1] + '\t' + item[2] + '\t' + str(area) + '\n')
                    
                    a = a+1

        # fdr p values 
        Bonferroni(p_value_bed_file, outpath, target)
        print('write %s BenjaminiFDR p-value to bed file' % target)
        bonferroni_path = os.path.join(outpath, 'bonferroni_%s_p_value.bed'%target)
        filtered_bonferroni_path = os.path.join(outpath, 'filtered_bonferroni_%s_p_value.bed' % target)
        bash_filter = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'filter_p_value.sh')
        filtered_predicted_file = os.path.join(outpath, 'positive_%s.bed' % target)

        if target == 'genebody':
            cmd_filter = ['bash', bash_filter, bonferroni_path, filtered_bonferroni_path, filtered_predicted_file, str(filter_threshold)]
        else:
            cmd_filter = ['bash', bash_filter, bonferroni_path, filtered_bonferroni_path, filtered_predicted_file, str(filter_threshold)]
        subprocess.call(cmd_filter)


def overlap_stat(outpath,
                 label_list,
                 target_list,
                 extend=50,
                 include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                 threshold_peak=0.6,
                 raw_genebody_file=None,
                 macs2=False,
                 ):
    
    eva_out = os.path.join(outpath, 'eva_results.txt')
    if os.path.exists(eva_out):
        os.remove(eva_out)

    for i in range(len(target_list)):
        target = target_list[i]
        print(target)
        label_path = label_list[i]
        predicted_path = os.path.join(outpath, 'positive_%s.bed' % target)

        if macs2:
            predicted_path = os.path.join(outpath, 'predicted_macs2_%s.bed' % target)

        # select base iou
        overlap_file = os.path.join(outpath, 'overlap_positive_%s.bed' % target)
        cmd_intersect = 'bedtools window -a ' + label_path + ' -b ' + predicted_path + ' -w %d | bedtools overlap -i stdin -cols 2,3,5,6  > ' % extend + overlap_file
        p = subprocess.Popen(cmd_intersect, shell=True)
        p.wait()
        
        # 与label有overlap的预测peak，用于计算iou
        overlap_peaks = read_peaks(overlap_file, include_chr=include_chr)

        ###################################
        #iou 计算
        ###################################
        pre_iou = 0
        # 带有iou值的文件，只包含iou大于某个阈值的overlap peaks中的条目
        iou_path = os.path.join(outpath, 'iou_positive_%s.bed' % target)
        peak = overlap_peaks[0]
        peak.append(str(peak_IOU(peak, extend=extend)))

        iou_peaks = [peak]

        for peak in overlap_peaks[1:]:
            iou = peak_IOU(peak, extend=extend)
            # 一个label对应多个预测peak时，把iou加起来，然后赋值给这些peak
            if peak[:3] == iou_peaks[-1][:3]:
                iou += float(pre_iou)
                i = -1
                while iou_peaks[i][:3] == peak[:3] and i >= 0:
                    iou_peaks[-1][-1] = str(iou)
                    i -= 1
            
            peak.append(str(iou))
            iou_peaks.append(peak)
            pre_iou = iou

        filter_peak = []
        # 将使用iou过滤后的peak写入iou_path文件，并记录在filter_peak中
        with open(iou_path, 'w') as w_obj:
            for peak in iou_peaks:
                if float(peak[-1]) >= threshold_peak:
                    filter_peak.append(peak)
                    # print(float(peak[-3]))
                    w_peak = ''
                    for item in peak:
                        w_peak += item + '\t'
                    w_peak = w_peak[:-1] + '\n'
                    w_obj.write(w_peak)

        # write true positive to bed
        true_positive_file = os.path.join(outpath, 'true_positive_%s.bed' % target)
        with open(true_positive_file, 'w') as w_obj:
            pre_peak = filter_peak[0]
            # 1-3 是label 4-5是predicted
            w_obj.write(pre_peak[3] + '\t' + pre_peak[4] + '\t' + pre_peak[5] + '\n')
            for peak in filter_peak[1:]:
                # remove duplicates
                if [peak[3], peak[4], peak[5]] == [pre_peak[3], pre_peak[4], pre_peak[5]]:
                    continue
                if float(peak[-1]) >= threshold_peak:
                    w_obj.write(peak[3] + '\t' + peak[4] + '\t' + peak[5] + '\n')
                pre_peak = peak

        #########################
        # false positive
        #########################
        # 计算没有match上的label和预测peak
        nonintersect_label_path = os.path.join(outpath, 'nonintersect_label_%s.bed' % target)
        nonintersect_predicted_peak = os.path.join(outpath, 'false_positive_%s.bed' % target)

        # false negative and false positive
        bash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tp_fp.sh')
        cmd_fn_fp = ['bash', bash_path, label_path, iou_path, nonintersect_label_path, nonintersect_predicted_peak, predicted_path]
        subprocess.call(cmd_fn_fp)
        # 只保留include chr的false positive
        write_peaks(nonintersect_predicted_peak, include_chr=include_chr)

        #########################
        # 计算tp fp
        #########################
        num_predicted_peak = len(read_peaks(predicted_path, include_chr=include_chr))
        fp = len(read_peaks(nonintersect_predicted_peak, include_chr=include_chr))

        tp = num_predicted_peak - fp

        # unrecorded genebody
        # 然后加上新match上的（非转录的基因、或者新匹配的转录本）
        if target == 'genebody':
            # 使用bedtools对预测peak和包括lincRNA和转录本的文件找overlap
            overlap_file = os.path.join(outpath, 'overlap_with_all_transcript_%s.bed' % target)
            cmd_intersect = 'bedtools window -a ' + raw_genebody_file + ' -b ' + nonintersect_predicted_peak + ' -w 50 | bedtools overlap -i stdin -cols 2,3,9,10  > ' + overlap_file
            p = subprocess.Popen(cmd_intersect, shell=True)
            p.wait()

            # 与label有overlap的预测peak，用于计算iou
            overlap_peaks = read_peaks(overlap_file, include_chr=include_chr)

            iou_path = os.path.join(outpath, 'iou_match_with_lincRNA_transcript_%s.bed' % target)
            iou_peaks = []
            # 记录每个转录本的iou
            for peak in overlap_peaks:
                iou = peak_IOU(peak, extend=extend)
                
                peak.append(str(iou))
                iou_peaks.append(peak)

            # 选择iou大于阈值的条目
            filted_peak = []
            with open(iou_path, 'w') as w_obj:
                for peak in iou_peaks:
                    if float(peak[-1]) >= threshold_peak:
                        filted_peak.append(peak)
                        # print(float(peak[-3]))
                        w_peak = ''
                        for item in peak:
                            w_peak += item + '\t'
                        w_peak = w_peak[:-1] + '\n'
                        w_obj.write(w_peak)

            # remove duplicates
            new_matched_label = []
            rm_duc_filter_peak = [filted_peak[0]]
            # (1)找出和预测结果match上的转录本对应的gene的信息，这里是先找第一个转录本对应的
            for over_peak in overlap_peaks:
                # 找出和预测结果match上的转录本对应的gene的信息
                if over_peak[3] == 'gene' and over_peak[5] == filted_peak[0][5]:
                    new_matched_label.append(over_peak[:3])

            for peak in filted_peak[1:]:
                # 对已预测出来的match上的peak去重
                if peak[7:10] not in rm_duc_filter_peak:
                    rm_duc_filter_peak.append(peak[7:10])
                    # (2)找出和预测结果match上的转录本对应的gene的信息
                    for over_peak in overlap_peaks:
                        if over_peak[3] == 'gene' and over_peak[5] == peak[5] and over_peak[:3] not in new_matched_label:
                            new_matched_label.append(over_peak[:3])

            # 将和转录本和LINC RNA匹配的genebody记录在文件中
            num_match_with_lincRNA_transcript = len(rm_duc_filter_peak)
            # 新匹配上的预测正例
            match_with_lincRNA_transcript_path = os.path.join(outpath, 'match_with_lincRNA_transcript_%s.bed' % target)
            with open(match_with_lincRNA_transcript_path, 'w') as w_obj:
                for peak in rm_duc_filter_peak:
                    w_obj.write("\t".join(peak) + "\n")
            # print(num_match_with_lincRNA)
            tp = tp + num_match_with_lincRNA_transcript
            fp = fp - num_match_with_lincRNA_transcript

        #########################
        # 计算label查全率
        #########################
        num_new_matched_ori_label = 0
        num_label_peak = len(read_peaks(label_path, include_chr=include_chr))
        nonintersect_label = read_peaks(nonintersect_label_path, include_chr=include_chr)
        num_nonintersect_label = len(nonintersect_label)
        if target == 'genebody':
            # print(num_label_peak)
            # 计算之前没有match的label现在match上了几个
            for label in nonintersect_label:
                if label in new_matched_label:
                    num_new_matched_ori_label += 1
            # print(len(new_matched_label))
            # non_label数量=原non_label数量 - 没匹配上整个gene但匹配了转录本，的数量
            num_nonintersect_label = num_nonintersect_label - num_new_matched_ori_label
            # 这里要不要加上括号里的这些（新匹配上的LINCRNA或者RO-seq不表达gene），存疑
            # 因为new match label还包含了linc rna，所有新增的应该是linc rna的数量: len(new_matched_label) - num_new_matched_ori_label
            num_intersect_linc_rna = len(new_matched_label) - num_new_matched_ori_label
            num_intersect_label = num_label_peak - num_nonintersect_label + num_intersect_linc_rna
        else:
            num_intersect_label = num_label_peak - num_nonintersect_label

        #########################
        # negative TN FN
        #########################
        # 使用预测结果的补集作为反例，每个染色体正例反例数量相差1
        negative_path = os.path.join(outpath, 'negative_%s.bed' % target)
        idx_file = os.path.join(outpath, 'genome.idx')
        cmd_subtract = 'bedtools subtract -a ' + idx_file + ' -b ' + predicted_path + ' > ' + negative_path
        p = subprocess.Popen(cmd_subtract, shell=True)
        p.wait()

        # 反例与label做overlap
        overlap_file = os.path.join(outpath, 'overlap_negative_%s.bed' % target)
        cmd_intersect = 'bedtools window -a ' + label_path + ' -b ' + negative_path + ' -w 50 | bedtools overlap -i stdin -cols 2,3,5,6  > ' + overlap_file
        p = subprocess.Popen(cmd_intersect, shell=True)
        p.wait()

        overlap_peaks = read_peaks(overlap_file, include_chr=include_chr)

        pre_iou = 0
        iou_path = os.path.join(outpath, 'iou_negative_%s.bed' % target)

        peak = overlap_peaks[0]
        peak.append(str(peak_IOU(peak, extend=extend)))

        iou_peaks = [peak]

        for peak in overlap_peaks[1:]:
            iou = peak_IOU(peak, extend=extend)
            if peak[:3] == iou_peaks[-1][:3]:

                iou += float(pre_iou)
                i = -1
                while iou_peaks[i][:3] == peak[:3] and i >= 0:
                    iou_peaks[-1][-1] = str(iou)
                    i -= 1
            
            peak.append(str(iou))
            iou_peaks.append(peak)
            pre_iou = iou

        filted_peak = []
        # iou大于阈值的被视为假反例FN
        with open(iou_path, 'w') as w_obj:
            for peak in iou_peaks:
                if float(peak[-1]) >= threshold_peak:
                    filted_peak.append(peak)
                    # print(float(peak[-3]))
                    w_peak = ''
                    for item in peak:
                        w_peak += item + '\t'
                    w_peak = w_peak[:-1] + '\n'
                    w_obj.write(w_peak)

        # FN
        # remove duplicates
        rm_duc_filter_peak = [filted_peak[0]]
        if target == 'genebody':
            # 如果是genebody的话，排除与转录本match的genebody的gene(这个gene与反例iou大，但是同时正例覆盖了转录本)
            for peak in filted_peak[1:]:
                # 除了不能重复，还不能在new_matched_label中
                if peak[3:6] != rm_duc_filter_peak[-1][3:6] and peak[:3] not in new_matched_label:
                    rm_duc_filter_peak.append(peak)
        else:
            # 不是genebody的话，只需要去重就行
            for peak in filted_peak[1:]:
                if peak[3:6] != rm_duc_filter_peak[-1][3:6]:
                    rm_duc_filter_peak.append(peak)
        fn = len(rm_duc_filter_peak)

        fn_file = os.path.join(outpath, 'false_negative_%s.bed' % target)
        with open (fn_file, 'w') as w_obj:
            for peak in rm_duc_filter_peak:
                w_obj.write("\t".join(peak[3:6]) + "\n")

        negatives = read_peaks(negative_path, include_chr=include_chr)

        tn = len(negatives) - fn


        print('TP:', tp)
        print('FP:', fp)
        print('TN:', tn)
        print('FN:', fn)
        print('intersect_label:', num_intersect_label)
        print('non_intersect_label:', num_nonintersect_label)

        acc, pre, recall, f_score = classified_eva(tn, fn, fp, tp)
        print('Acc:', acc)
        print('Pre:', pre)
        print('Recall:', recall)
        print('F-Score:', f_score)
        print('label recall:', num_intersect_label / (num_intersect_label + num_nonintersect_label))
        if target == 'genebody':
            print('linc rna: ', num_intersect_linc_rna)
            print('rm linc label recall: ', (num_intersect_label - num_intersect_linc_rna) / (num_intersect_label + num_nonintersect_label - num_intersect_linc_rna))

        with open(eva_out, 'a') as w_obj:
            w_obj.write(target + '\n')
            w_obj.write('TP: %f\n' % tp)
            w_obj.write('FP: %f\n' % fp)
            w_obj.write('TN: %f\n' % tn)
            w_obj.write('FN: %f\n' % fn)
            w_obj.write('intersect_label: %f\n' % num_intersect_label)
            w_obj.write('non_intersect_label: %f\n' % num_nonintersect_label)

            w_obj.write('Acc: %.4f\n' % acc)
            w_obj.write('Pre: %.4f\n' % pre)
            w_obj.write('Recall: %.4f\n' % recall)
            w_obj.write('F-Score: %.4f\n' % f_score)
            w_obj.write('label recall: %.4f\n' % (num_intersect_label / (num_intersect_label + num_nonintersect_label)))
            if target == 'genebody':
                w_obj.write('linc rna: %.4f\n' % num_intersect_linc_rna)
                w_obj.write('rm linc label recall: %.4f\n' % ((num_intersect_label - num_intersect_linc_rna) / (num_intersect_label + num_nonintersect_label - num_intersect_linc_rna)))

    cmd_rm = ['rm', '-f', overlap_file]
    subprocess.call(cmd_rm)

    cmd_rm = ['rm', '-f', iou_path]
    subprocess.call(cmd_rm)


def peak_p_value(bw_list,
                 outpath,
                 target_list,
                 label_list,
                 threshold=0.142,
                 seq_length=114688,
                 window_size=128,
                 include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                 raw_genebody_file=None,
                 corrvar_matrix=None,
                 mean=None,
                 macs2=False,
                 bonferroni_direct=False
                 ):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    whole_genome_size = bw_2_chromosome_size(bw_file=bw_list[0], outpath=outpath, include_chr=include_chr)

    # bw to peak bw (0 or 1)
    # Positive
    bw_to_peak_bw = bw_2_peak_bed(bw_list, outpath, whole_genome_size, threshold, target_list, seq_length, window_size=window_size, include_chr=include_chr)
    print('bw to peak is completed')

    add_p_value = peak_add_p_value(target_list, bw_list, outpath, corrvar_matrix=corrvar_matrix, mean=mean, bonferroni_direct=bonferroni_direct)

    overlap_stat(outpath, label_list, target_list, extend=50, include_chr=include_chr, raw_genebody_file=raw_genebody_file, macs2=macs2)


def fe_classfication_evaluation(
                        pred_promoter,
                        pred_polya,
                        pred_insulator,
                        pred_genebody,
                        label_promoter,
                        label_polya,
                        label_insulator,
                        label_genebody,
                        raw_genebody_file,
                        outdir='.',
                        include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                            'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                            'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
                        ):
    """
    evaluate classfication performance.

    Args:
        raw_genebody_file: bed file include coding and non-coding gene
    """
    target_list = ['promoter',  'polya', 'insulator', 'genebody']

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    label_list = [label_promoter, label_polya, label_insulator, label_genebody,]

    bw_list = [pred_promoter, pred_polya, pred_insulator, pred_genebody]

    ###################################
    # 1. read depth statistics
    ##################################
    # must be promoter

    bw_file = bw_list[0]
    read_depth_out = os.path.join(outdir, 'promoter.readDepth.bed')
    print('start read file: %s' % bw_file)
    get_read_depth(bw_file, read_depth_out)
    print('complete file: %s read'  % bw_file)
    time.sleep(10)

    print('removing insignifice region ...')
    rm_meaningless_region(read_depth_out)
    print('completed insignifice region ...')
    
    ###################################
    # 2. cov
    ##################################
    mean, std, cov_mat = mean_and_cov(read_depth_out)
    print('mean: ', mean)
    print('std: ', std)
    print('covarience matrix: \n', cov_mat)

    ###################################
    # 3. slicing threshold
    ##################################
    threshold = round(np.e ** norm.interval(0.99, loc=mean, scale=std)[1], 2)

    mean = np.asarray([mean] * 5)
    print('threshold ', threshold)

    ###################################
    # 4. bw slice
    ##################################
    out_eva = os.path.join(outdir, 'tre_evaluation')
    if not os.path.exists(out_eva):
        os.mkdir(out_eva)

    whole_genome_size = bw_2_chromosome_size(bw_file=bw_list[0], outpath=out_eva, include_chr=include_chr)

    bw_to_peak_bw = bw_2_peak_bed(bw_list,
                                out_eva,
                                whole_genome_size,
                                threshold,
                                target_list,
                                seq_length=114688,
                                window_size=128,
                                include_chr=include_chr
                                )
    print('bw to peak is completed')

    ###################################
    # 5. futher slice genebody, smooth and slice
    ##################################
    # smooth
    chr_length_file = os.path.join(out_eva, 'genome.length')
    print('smoothing genebody ...')
    smooth_bw(bw_list[3], out_eva, chr_length_file)

    # # # # slice
    print('slicing genebody ...')
    smoothed_bw_file = os.path.join(out_eva, 'smooth_genebody.bw')
    slice_peak(out_eva, smoothed_bw_file)

    ###################################
    # 6. add p-value
    ##################################
    add_p_value = peak_add_p_value(target_list,
                                bw_list,
                                out_eva,
                                corrvar_matrix=cov_mat,
                                mean=mean,
                                bonferroni_direct=False,
                                filter_threshold=1e-3,
                                )
    
    ###################################
    # 7. evaluation
    ##################################
    overlap_stat(out_eva,
                label_list,
                target_list,
                extend=50,
                include_chr=include_chr,
                raw_genebody_file=raw_genebody_file,
                macs2=False)



if __name__ == '__main__':

    # Useage:
    fe_classfication_evaluation(
                        pred_promoter='/local/zzx/code/BioSeq2Seq/test_samples/fe/promoter.bw',
                        pred_polya='/local/zzx/code/BioSeq2Seq/test_samples/fe/polya.bw',
                        pred_insulator='/local/zzx/code/BioSeq2Seq/test_samples/fe/insulator.bw',
                        pred_genebody='/local/zzx/code/BioSeq2Seq/test_samples/fe/genebody.bw',
                        label_promoter='/local/zzx/code/BioSeq2Seq/genome_regions/FE_file/K562/all_promoter.bed',
                        label_polya='/local/zzx/code/BioSeq2Seq/genome_regions/FE_file/K562/polya.bed',
                        label_insulator='/local/zzx/code/BioSeq2Seq/genome_regions/FE_file/K562/insulator.bed',
                        label_genebody='/local/zzx/code/BioSeq2Seq/genome_regions/FE_file/K562/genebody.bed',
                        raw_genebody_file='/local/zzx/code/BioSeq2Seq/test_samples/fe/genebody_7_lines_raw.bed',
                        outdir='/local/zzx/code/BioSeq2Seq/test_samples/fe/out',
                        include_chr=['chr22']
    )

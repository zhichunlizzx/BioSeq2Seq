#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from operator import itemgetter
import subprocess
from dataloader import get_dataset
import tensorflow as tf
import numpy as np
import pysam
import math
import pyBigWig
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def check_if_out_of_bounds(samples, chrom_size):
    """
    Check for out of bounds samples

    Args:
        samples: a data frame of samples
        chrom_size: chromosize of the reference genome of samples
    
    Return:
        The chromosome of the false sample or None
    """
    samples = samples[np.argsort(samples, axis=0)[:, 0]]

    for chr in np.unique(samples[:, 0]):
        chr_idx = np.argwhere(samples[:, 0] == chr).squeeze(-1)
        chr_samples = samples[chr_idx]
        chr_samples = chr_samples[np.argsort(chr_samples[:, -1].astype(int))]
        if int(chr_samples[0][1]) < 0 or int(chr_samples[-1][2]) > chrom_size[chr][0][-1]:
            return chr
        
    return None


def bw_2_chromosome_size(bw_file, outdir=None):
    """Read chromosome size from .bw file"""
    try:
        bw_open = pyBigWig.open(bw_file)
    except:
        raise Exception('Error: bw_file must be a bigwig file')
    
    chromsize = bw_open.chroms()

    if outdir is not None:
        reference_genome_idx = os.path.join(outdir, 'idx.fai')
        with open(reference_genome_idx, 'w') as w_obj:
            for chr in chromsize:
                w_obj.write(chr + '\t' + str(chromsize[chr]) + '\n')
                chromsize[chr] = [(0, chromsize[chr])]
    else:
        for chr in chromsize:
            chromsize[chr] = [(0, chromsize[chr])]
    return chromsize


def fai_2_choromosome_size(fai_file):
    """Read chromosome size from fai file"""
    with open(fai_file, 'r') as r_obj:
        lines = r_obj.readlines()
    sections = [section.split() for section in lines]

    chrom_size = {}
    for section in sections:
        chrom_size[section[0]] = [(0, int(section[1]))]
    
    return chrom_size


def load_chromosomes(genome_file):
    """ Load genome segments from either a FASTA file or chromosome length table. """
    # is genome_file FASTA or (chrom,start,end) table?
    file_fasta = (open(genome_file).readline()[0] == '>')

    chrom_segments = {}
    try:
        if file_fasta:
            fasta_open = pysam.Fastafile(genome_file)
            for i in range(len(fasta_open.references)):
                chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
            fasta_open.close()
        else:
            # (chrom,start,end) table
            for line in open(genome_file):
                a = line.split()
                chrom_segments[a[0]] = [(0, int(a[1]))]
    except:
        raise Exception('Error: reference genome file errore')

    return chrom_segments

def write_predicted_result(
                        results,
                        out_path,
                        chr_length,
                        target_list,
                        reference_genome_idx,
                        seq_length=114688,
                        window_size=128,
                        pre_out='BioSeq2Seq',
                        ):
    """ 
    Write result to bigwig file

    Args:
        results: predicted result, {chr:[{start:xx, end:xx, result:xx}]}
        out_path: output path
        chr_length: chromosome length
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
                # assign 0 to the area not covered by the sample
                if chr_result[0]['start'] > 0:
                    w_obj.write(chr + '\t' + str(0) + '\t' + str(chr_result[0]['start']) + '\t' + str(0) + '\n')

                # write predict result
                last_end = 0
                for item in chr_result:
                    if item['start'] >= last_end: 
                        for i in range(target_length):
                            start = item['start'] + i * window_size
                            end = start + window_size
                            w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                    else:
                        print(item)
                        gap_h = last_end - item['start']
                        h_start = gap_h // window_size
                        w_obj.write(chr + '\t' + str(last_end) + '\t' + str(item['start'] + window_size * (h_start+1)) + '\t' + str(item['predicted'][h_start][j]) + '\n')
                        for i in range(h_start+1, target_length):
                            start = item['start'] + i * window_size
                            end = start + window_size 
                            w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                last_end = item['end']

                # assign 0 to the area not covered by the sample
                if chr_result[-1]['end'] < chr_length[chr]:
                    w_obj.write(chr + '\t' + str(chr_result[-1]['end']) + '\t' + str(chr_length[chr]) + '\t' + str(0) + '\n')

    # bedgraph to bigwig
    for j in range(len(target_list)):
        bed_path = os.path.join(out_path, target_list[j] + '.bedgraph')
        bedgraph_path_sorted = os.path.join(out_path, target_list[j] + '_sorted.bedgraph')
        cmd_bedSort = 'sort-bed ' + bed_path + ' > ' + bedgraph_path_sorted
        p = subprocess.Popen(cmd_bedSort, shell=True)
        p.wait()

        bw_path = os.path.join(out_path, pre_out + '-' + target_list[j] + '.bw')

        cmd = ['bedGraphToBigWig', bedgraph_path_sorted, reference_genome_idx, bw_path]
        subprocess.call(cmd)

        cmd_rm = ['rm', '-f', bed_path]
        subprocess.call(cmd_rm)

        cmd_rm = ['rm', '-f', bedgraph_path_sorted]
        subprocess.call(cmd_rm)

    return True


def predicted_to_bigwig(
                        model,
                        samples,
                        reference_genome_file,
                        sequencing_data_file,
                        target_list,
                        chrom_size,
                        out_path,
                        data_type='dna+seq',
                        extend=40960,
                        nan=0,
                        seq_length=114688,
                        window_size=128,
                        model_type='HistoneModification',
                        pre_out='BioSeq2Seq',
                        ):
    """ 
    Write result to bigwig file

    Args:
        model: trained model
        samples: samples with length of 114688 bp, [num_of_samples, 3]
        reference_genome_file: reference genome file
        sequencing_data_file: file path of sequcing data
        target_list: target sequencing data list
        chrom_size: chromosize of the reference genome of samples
        out_path: output path
        data_type: the data type of the input data of model
        extend: the length extended on both sides of each sample in order to take full advantage of the transformer
        nan: replace 'Nan' or 'Inf' in the data with the parameter value

    Return:
        None
    """
    @tf.function
    def predict(data):
        return model(data, is_training=False)
    
    results = {}
    print(target_list)
    # chromosome length
    chr_length = {}
    for chr in np.unique(samples[:, 0]):
        results[chr] = []
        chr_length[chr] = chrom_size[chr][0][1]

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # chromosome length file
    reference_genome_idx = os.path.join(out_path, 'idx.fai')
    with open(reference_genome_idx, 'w') as w_obj:
        for chr in chrom_size:
            w_obj.write(chr + '\t' + str(chrom_size[chr][0][1]) + '\n')
    
    test_dataset = get_dataset(samples, reference_genome_file, sequencing_data_file, data_type=data_type, extend=extend, nan=nan, model_type=model_type).batch(1)

    # record results
    for j, data in tqdm(enumerate(test_dataset)):
        result = {}
        predicted_tf = predict(data)
        
        result['chr'] = samples[j][0]
        result['start'] = int(samples[j][1])
        result['end'] = int(samples[j][2])
        result['predicted'] = predicted_tf[0].numpy()
        results[result['chr']].append(result)
        # print(data)
        # print(predicted_tf)
        # print(result['chr'], result['start'], result['end'])

    write_down = write_predicted_result(results, out_path, chr_length, target_list, reference_genome_idx, seq_length=seq_length, window_size=window_size, pre_out=pre_out)

    os.remove(reference_genome_idx)
        
    return True


def _reduced_shape(shape, axis):
    if axis is None:
        return tf.TensorShape([])
    return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


def split_based_chr(samples, divide_chr=['chr22']):
    '''
    split samples to training, validation and test set

    Args:
        samples: [num_samples, 3]
        divide_chr: select the samples of chromosomes in divide_chr

    Return:
        samples_divided: the samples of chromosomes in divide_chr
        samples_reserved: the rest of the samples
    '''
    divided_idx = [sample in divide_chr for sample in samples[:, 0]]

    reserved_idx = (np.asarray(divided_idx) == False)

    samples_reserved = samples[reserved_idx]
    samples_divided = samples[divided_idx]
    
    return samples_divided, samples_reserved


def split_based_percent(samples, chose_sample_percent=1.):
    '''
    split samples to two part based on appointed percent

    Args:
        samples: [num_samples, 3]
        chose_sample_percent: division ratio(float)

    Return:
        chose_samples: the sample of the chosen_sample_percent ratio in samples
        reserved_samples: the sample of the (1 - chosen_sample_percent) ratio in samples
    '''

    if chose_sample_percent > 1:
        raise Exception('Error: chose_sample_percent must be an integer less than 1')

    num_chose_sample = math.floor(chose_sample_percent * len(samples))

    chose_sample_idx = list(np.random.choice(list(range(len(samples))), num_chose_sample, replace=False))

    reserved_sample_idx = list(set(list(range(len(samples)))).difference(set(chose_sample_idx)))

    chose_samples = samples[chose_sample_idx]
    reserved_samples = samples[reserved_sample_idx]

    return chose_samples, reserved_samples


def split_based_num(samples, chose_num=1):
    '''
    split samples to two part based on num of samples

    Args:
        samples: [num_samples, 3]
        chose_num: chose num

    Return:
        chose_samples: the sample of the chosen_sample_percent ratio in samples
        reserved_samples: the sample of the (1 - chosen_sample_percent) ratio in samples
    '''
    # select a part of the sample
    # train and valid
    if chose_num < 1 or not(type(chose_num)==int):
        raise Exception('Error: chose_sample_num must be an integer greater than 0')
    
    if len(samples) < chose_num:
        raise Exception('Error: chose_num exceeds the maximum num of samples')

    num_chose_sample = chose_num

    chose_sample_idx = list(np.random.choice(list(range(len(samples))), num_chose_sample, replace=False))

    reserved_sample_idx = list(set(list(range(len(samples)))).difference(set(chose_sample_idx)))

    chose_samples = samples[chose_sample_idx]
    reserved_samples = samples[reserved_sample_idx]

    return chose_samples, reserved_samples

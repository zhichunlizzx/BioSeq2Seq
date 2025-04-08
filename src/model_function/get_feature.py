#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import numpy as np
import pyBigWig
from utils.dna_io import dna_1hot
import pysam
from utils.genome_tools import CovFace


def fetch_dna(fasta_open, chrm, start, end):
    """
    Fetch DNA when start/end may reach beyond chromosomes.
    
    Args:
        fasta_open: an object of the opened reference genome file
        chrm: chromosome
        start: start of the sample
        end: end of the sample
    
    Return:
        seq_dna: the onehot encoding of the sample, [seq_length, 4]
    
    """

    # initialize sequence
    seq_len = end - start
    seq_dna = ''

    # add N's for left over reach
    if start < 0:
        seq_dna = 'N'*(-start)
        start = 0

    # get dna
    seq_dna += fasta_open.fetch(chrm, start, end)

    # add N's for right over reach
    if len(seq_dna) < seq_len:
        seq_dna += 'N'*(seq_len-len(seq_dna))

    return seq_dna


def get_double_stranded_input_feature(samples, sequencing_data_files, extend=40960, nan=None):
    """ 
    get ground truth of samples

    Args:
        samples: samples with length of 114688 bp, [num_of_samples, 4]
        sequencing_data_files: the path of sequencing bigwig files
        extend: the length extended to take advantage of the Transformer
        nan: replace outliers with parameter values
        
    Return:
        features: [num_of_samples, 196608, 3]
    """

    # read sequencing data
    genome_cov_file_minus = sequencing_data_files[0]
    genome_cov_file_plus = sequencing_data_files[1]

    try:
        genome_cov_open_minus = CovFace(genome_cov_file_minus)
        genome_cov_open_plus = CovFace(genome_cov_file_plus)
    except:
        raise Exception('there is a error when reading:',sequencing_data_files)
    
    samples_feature = []
    for sample in samples:
        chr, start, end = sample[0], int(sample[1]) - extend, int(sample[2]) + extend
        chr_length = pyBigWig.open(genome_cov_file_minus).chroms()

        p_start = start if start > 0 else 0
        p_end = end if end < chr_length[chr] else chr_length[chr]

        try:
            seq_cov_nt_minus = genome_cov_open_minus.read(chr, p_start, p_end)
            seq_cov_nt_plus = genome_cov_open_plus.read(chr, p_start, p_end)
        except:
            raise Exception('There may be an out-of-bounds error in %s, start:%s, end:%s'%(chr, start, end))

        baseline_cov_minus = np.percentile(seq_cov_nt_minus, 100*0.5)
        baseline_cov_plus = np.percentile(seq_cov_nt_plus, 100*0.5)
        if nan is None:
            baseline_cov_minus = np.nan_to_num(baseline_cov_minus)
            baseline_cov_plus = np.nan_to_num(baseline_cov_plus)
        else:
            baseline_cov_minus = nan
            baseline_cov_plus = nan

        # NaN to value
        nan_mask_minus = np.isnan(seq_cov_nt_minus)
        nan_mask_plus = np.isnan(seq_cov_nt_plus)
        seq_cov_nt_minus[nan_mask_minus] = baseline_cov_minus
        seq_cov_nt_plus[nan_mask_plus] = baseline_cov_plus

        # Inf to value
        inf_mask_minus = np.isinf(seq_cov_nt_minus)
        inf_mask_plus = np.isinf(seq_cov_nt_plus)
        seq_cov_nt_minus[inf_mask_minus] = baseline_cov_minus
        seq_cov_nt_plus[inf_mask_plus] = baseline_cov_plus

        # assign values to parts outside the genome range
        seq_cov_nt_minus = np.hstack((np.zeros(abs(start-p_start)), seq_cov_nt_minus))
        seq_cov_nt_minus = np.hstack((seq_cov_nt_minus, np.zeros(abs(end-p_end)))).astype('float16')
        seq_cov_nt_plus = np.hstack((np.zeros(abs(start-p_start)), seq_cov_nt_plus))
        seq_cov_nt_plus = np.hstack((seq_cov_nt_plus, np.zeros(abs(end-p_end)))).astype('float16')

        # the value of the third channel
        seq_cov_minus_plus = abs(seq_cov_nt_minus) + abs(seq_cov_nt_plus)

        samples_feature.append([abs(seq_cov_nt_minus), seq_cov_nt_plus, seq_cov_minus_plus])
 
    # [N, C, L] -> [N, L, C]
    samples_feature = np.asarray(samples_feature, dtype='float32').transpose(0, 2, 1)

    return samples_feature


def get_single_stranded_input_feature(samples, sequencing_data_file, extend=40960, nan=None):
    """ 
    get ground truth of samples

    Args:
        samples: samples with length of 114688 bp, [num_of_samples, 4]
        sequencing_data_file: the path of the sequencing bigwig file
        extend: the length extended to take advantage of the Transformer
        nan: replace outliers with parameter values

    Return:
        features: [num_of_samples, 196608, 1]
    """
    # read sequencing data
    genome_cov_file = sequencing_data_file[0]
    chr_length = pyBigWig.open(genome_cov_file).chroms()
    try:
        genome_cov_open = CovFace(genome_cov_file)
    except:
        raise Exception('there is a error when reading:', sequencing_data_file[0])

    samples_feature = []

    for sample in samples:
        chr, start, end = sample[0], int(sample[1]) - extend, int(sample[2]) + extend
        p_start = start if start > 0 else 0
        p_end = end if end < chr_length[chr] else chr_length[chr]

        try:
            seq_cov_nt = genome_cov_open.read(chr, p_start, p_end)
        except:
            raise Exception('There may be an out-of-bounds error in %s, start:%s, end:%s'%(chr, start, end))

        if nan is None:
            baseline_cov = np.percentile(seq_cov_nt, 100*0.5)
            baseline_cov = np.nan_to_num(baseline_cov)
        else:
            baseline_cov = nan

        # NaN to value
        nan_mask = np.isnan(seq_cov_nt)
        seq_cov_nt[nan_mask] = baseline_cov

        # Inf to value
        inf_mask = np.isinf (seq_cov_nt)
        seq_cov_nt[inf_mask] = baseline_cov

        # assign values to parts outside the genome range
        seq_cov_nt = np.hstack((np.zeros(abs(start-p_start)), seq_cov_nt))
        seq_cov_nt = np.hstack((seq_cov_nt, np.zeros(abs(end-p_end)))).astype('float16')

        samples_feature.append([seq_cov_nt])
    
    # [N, C, L] -> [N, L, C]
    samples_feature = np.asarray(samples_feature, dtype='float32').transpose(0, 2, 1)
    return samples_feature


def get_target_feature(samples, sequencing_data_files, window_width=128, mean_method='mean', nan=None):
    """ 
    get ground truth data of every samples

    Args:
        samples: samples with length of 114688 bp, [num_of_samples, 4]
        sequencing_data_files: the path of the ground truth or label file
        window_width: resolution ratio, represents a window as a point
        mean_method: representation method of data in the window
        nan: replace outliers with parameter values

    Return:
        samples_feature: [num_of_samples, 896, 1]
    """
    samples_feature = []
    seq_length = int(samples[0][2]) - int(samples[0][1])
    target_length = seq_length // window_width
    for sample in samples:
        chr, start, end = sample[0], int(sample[1]), int(sample[2])
        target_seq_covs = []
        for file in sequencing_data_files:
            
            try:                                                                                                                                                                                                                                                                                                                                                                                                                                     
                genome_cov_open = CovFace(file)
            except:
                raise Exception('there is a error when reading:', file)

            try:
                seq_cov_nt = genome_cov_open.read(chr, start, end)
            except:
                raise Exception('read error %s in chromosome: , %s, %s' % (file, chr, start, end))
                
      
            # NaN to value
            if nan is None:
                baseline_cov = np.percentile(seq_cov_nt, 100*0.5)
                baseline_cov = np.nan_to_num(baseline_cov)
            else:
                baseline_cov = nan

            # baseline_cov = np.percentile(seq_cov_nt, 100*0.5)
            # baseline_cov = np.nan_to_num(baseline_cov)

            nan_mask = np.isnan(seq_cov_nt)
            seq_cov_nt[nan_mask] = baseline_cov

            # Inf to value
            inf_mask = np.isinf(seq_cov_nt)
            seq_cov_nt[inf_mask] = baseline_cov

            # pool
            seq_cov = seq_cov_nt.reshape(target_length, window_width)
            if mean_method == 'mean':
                seq_cov = seq_cov.mean(axis=1, dtype='float32')
            elif mean_method == 'sum':
                seq_cov = seq_cov.sum(axis=1, dtype='float32')
            elif mean_method == 'median':
                seq_cov = seq_cov.median(axis=1, dtype='float32')
            elif mean_method == 'max':
                seq_cov = seq_cov.max(axis=1)
            seq_cov = np.asarray(seq_cov, dtype='float16')
            seq_cov = np.clip(seq_cov, -384.0, 384.0)
            target_seq_covs.append(seq_cov)
        
        samples_feature.append(target_seq_covs)
  
    # [N, C, L] -> [N, L, C]
    samples_feature = np.asarray(samples_feature, dtype='float32').transpose(0, 2, 1)
    return samples_feature


def get_dna_seq_onehot_encoding(samples, dna_fasta_file, extend=40960):
    """ 
    get ground truth of samples

    Args:
        samples: samples with length of 114688 bp, [num_of_samples, 4]
        dna_fasta_file: the path of the reference genome date file
        extend: the length extended to take advantage of the Transformer

    Return:
        samples_feature: onehot encoding for A T C G, [num_of_samples, 196608, 4]
    """
    # open FASTA
    fasta_open = pysam.Fastafile(dna_fasta_file)
    
    onehot_encodings = []
    for sample in samples:
        chr, start, end = sample[0], int(sample[1]) - extend, int(sample[2]) + extend

        # read FASTA
        seq_dna = fetch_dna(fasta_open, chr, start, end)

        # one hot code (N's as zero)
        seq_1hot = dna_1hot(seq_dna, n_uniform=False, n_sample=False)

        onehot_encodings.append(seq_1hot.astype(float))

    return np.asarray(onehot_encodings, dtype='float32')


def get_input_seq_feature(samples, sequencing_data_files, extend=40960, nan=None):
    """ 
    get ground truth of samples

    Args:
        samples: samples with length of 114688 bp, [num_of_samples, 4]
        sequencing_data_files: the path of the sequencing bigwig file
        extend: the length extended on both sides of each sample in order to take full advantage of the transformer
        nan: replace 'Nan' or 'Inf' in the data with the parameter value

    Return:
        features: [num_of_samples, 196608, 1]
    """
    if len(sequencing_data_files) == 2:
        return get_double_stranded_input_feature(samples, sequencing_data_files, extend=extend, nan=nan)
    elif len(sequencing_data_files) == 1:
        return get_single_stranded_input_feature(samples, sequencing_data_files, extend=extend, nan=nan)
    else:
        raise Exception('Error: plese provide the correct seq type(single or double).')

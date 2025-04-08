#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model_function.get_feature import get_input_seq_feature, get_target_feature, get_dna_seq_onehot_encoding
import tensorflow as tf
import numpy as np


def get_dna_seq_features(samples, reference_genome_file, sequencing_data_file, target_sequencing_file, window_width=128, extend=40960, nan=None):
    """"
    get the data of the one-hot encoding of DNA sequence, the input sequencing data and the output sequencing data(gound truth)
    
    Args:
        samples: sample list [num of samples, 3]
        reference_genome_file: the path of the reference genome date file
        sequencing_data_file: the path of the sequencing bigwig file
        target_sequencing_file: the path of the ground truth or label file
        window_width: the genomic signal within the window of length window_width will be represented as a value
        extend: the length extended on both sides of each sample in order to take full advantage of the transformer
        nan: replace 'Nan' or 'Inf' in the data with the parameter value

    Return:
        sample_gen: generator
        sample_types: type of the generated data
        sample_shapes: shape of the generated data
    """

    # generater
    def sample_gen():
        for sample in samples:
            seq_num = int(sample[3]) 
            # random_seq=0                                                                                                      
            sample = [[sample[0], sample[1], sample[2]]]
            
            dna_encoding = np.squeeze(get_dna_seq_onehot_encoding(sample, reference_genome_file, extend=extend), 0)
            seq_feature = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[0][seq_num], extend=extend, nan=nan), 0)
            target = np.squeeze(get_target_feature(sample, target_sequencing_file, window_width=window_width, mean_method='mean', nan=nan), 0)

            _ = yield (dna_encoding, seq_feature, target)

    
    sample_types = (tf.float32, tf.float32, tf.float32)
    # dimension of seq feature
    dim_seq = 2 ** (len(sequencing_data_file[0][0])) - 1
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (DNA, seq, ground_truth)
    sample_shapes = (
                    (seq_length + 2 * extend, 4),
                    (seq_length + 2 * extend, dim_seq),
                    (int(seq_length / window_width), len(target_sequencing_file))
                    )

    return sample_gen, sample_types, sample_shapes


def get_seq_seq_features(samples, reference_genome_file, sequencing_data_file, target_sequencing_file, window_width=128, extend=40960, nan=None):
    """get the data of the first input sequencing data, the second input sequencing data and the output sequencing data(gound truth)"""

    def sample_gen():
        for sample in samples:
            seq_num = int(sample[3]) 
            sample = [[sample[0], sample[1], sample[2]]]

            seq_feature_1 = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[0][seq_num], extend=extend, nan=nan), 0)
            seq_feature_2 = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[1][seq_num], extend=extend, nan=nan), 0)
            target = np.squeeze(get_target_feature(sample, target_sequencing_file, window_width=window_width, mean_method='mean', nan=nan), 0)

            _ = yield (seq_feature_1, seq_feature_2, target)

    
    sample_types = (tf.float32, tf.float32, tf.float32)
    # dimension of seq feature
    dim_seq_1 = 2 ** (len(sequencing_data_file[0][0])) - 1
    dim_seq_2 = 2 ** (len(sequencing_data_file[1][0])) - 1
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (seq_1, seq_2, ground_truth)
    sample_shapes = (
                    (seq_length + 2 * extend, dim_seq_1),
                    (seq_length + 2 * extend, dim_seq_2),
                    (int(seq_length / window_width), len(target_sequencing_file))
                    )

    return sample_gen, sample_types, sample_shapes


def get_dna_features(samples, reference_genome_file, sequencing_data_file, target_sequencing_file, window_width=128, extend=40960, nan=None, valid=False):
    """get the data of the one-hot encoding of DNA sequence and the output sequencing data(gound truth)"""

    def sample_gen():
        for sample in samples:
            sample = [[sample[0], sample[1], sample[2]]]
            dna_encoding = np.squeeze(get_dna_seq_onehot_encoding(sample, reference_genome_file, extend=extend), 0)
            target = np.squeeze(get_target_feature(sample, target_sequencing_file, window_width=window_width, mean_method='mean', nan=nan), 0)
            _ = yield (dna_encoding, target)
    
    sample_types = (tf.float32, tf.float32)
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (DNA, ground_truth)
    sample_shapes = (
                    (seq_length + 2 * extend, 4),
                    (int(seq_length / window_width), len(target_sequencing_file))
                    )

    return sample_gen, sample_types, sample_shapes


def get_seq_features(samples, reference_genome_file, sequencing_data_file, target_sequencing_file, window_width=128, extend=40960, nan=None):
    """get the data of the input sequencing data and the output sequencing data(gound truth)"""

    def sample_gen():
        
        for sample in samples:
            seq_num = int(sample[3]) 
            sample = [[sample[0], sample[1], sample[2]]]

            seq_feature_1 = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[0][seq_num], extend=extend, nan=nan), 0)
            target = np.squeeze(get_target_feature(sample, target_sequencing_file, window_width=window_width, mean_method='mean', nan=nan), 0)
            _ = yield (seq_feature_1, target)

    sample_types = (tf.float32, tf.float32)
    # dimension of seq feature
    dim_seq_1 = 2 ** (len(sequencing_data_file[0][0])) - 1
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (seq, ground_truth)
    sample_shapes = (
                    (seq_length + 2 * extend, dim_seq_1),
                    (int(seq_length / window_width), len(target_sequencing_file))
                    )

    return sample_gen, sample_types, sample_shapes


def get_predict_dna_seq(samples, reference_genome_file, sequencing_data_file, extend=40960, nan=None):
    """get the data of the one-hot encoding of DNA sequence and the input sequencing data"""

    def sample_gen():
        for sample in samples:
            sample = [[sample[0], sample[1], sample[2]]]
            dna_encoding = np.squeeze(get_dna_seq_onehot_encoding(sample, reference_genome_file, extend=extend), 0)
            seq_feature = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[0][0], extend=extend, nan=nan), 0)
            a = yield (dna_encoding, seq_feature)

    
    sample_types = (tf.float32, tf.float32)
    # dimension of seq feature
    dim_seq = 2 ** (len(sequencing_data_file[0][0])) - 1
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (DNA, seq)
    sample_shapes = ((seq_length + 2 * extend, 4), (seq_length + 2 * extend, dim_seq))
    return sample_gen, sample_types, sample_shapes


def get_predict_seq_seq(samples, reference_genome_file, sequencing_data_file, extend=40960, nan=None):
    """get the data of the first input sequencing data and the second input sequencing data"""

    def sample_gen():
        for sample in samples:
            sample = [[sample[0], sample[1], sample[2]]]
            seq_feature_1 = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[0][0], extend=extend, nan=nan), 0)
            seq_feature_2 = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[1][0], extend=extend, nan=nan), 0)
            a = yield (seq_feature_1, seq_feature_2)

    
    sample_types = (tf.float32, tf.float32)
    # dimension of seq feature
    dim_seq_1 = 2 ** (len(sequencing_data_file[0][0])) - 1
    dim_seq_2 = 2 ** (len(sequencing_data_file[1][0])) - 1
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (seq_1, seq_2)
    sample_shapes = ((seq_length + 2 * extend, dim_seq_1), (seq_length + 2 * extend, dim_seq_2))
    return sample_gen, sample_types, sample_shapes


def get_predict_dna(samples, reference_genome_file, sequencing_data_file, extend=40960, nan=None):
    """get the data of the one-hot encoding of DNA sequence"""

    def sample_gen():
        for sample in samples:
            sample = [[sample[0], sample[1], sample[2]]]
            dna_encoding = np.squeeze(get_dna_seq_onehot_encoding(sample, reference_genome_file, extend=extend), 0)
            a = yield (dna_encoding)

    
    sample_types = (tf.float32)
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (DNA)
    sample_shapes = ((seq_length + 2 * extend, 4))
    return sample_gen, sample_types, sample_shapes


def get_predict_seq(samples, reference_genome_file, sequencing_data_file, extend=40960, nan=None):
    """get the data of the input sequencing data"""

    def sample_gen():
        for sample in samples:
            sample = [[sample[0], sample[1], sample[2]]]
            seq_feature = np.squeeze(get_input_seq_feature(sample, sequencing_data_file[0][0], extend=extend, nan=nan), 0)
            a = yield (seq_feature)

    sample_types = (tf.float32)
    # dimension of seq feature
    dim_seq = 2 ** (len(sequencing_data_file[0][0])) - 1
    # length of the sample region
    seq_length = int(samples[0][2]) - int(samples[0][1])
    # (seq)
    sample_shapes = ((seq_length + 2 * extend, dim_seq))
    return sample_gen, sample_types, sample_shapes


def data_func(dna_encoding, seq_feature, target):
    """convert to tensor"""
    dna_encoding = tf.convert_to_tensor(dna_encoding, tf.float32)
    seq_feature = tf.convert_to_tensor(seq_feature, tf.float32)
    target = tf.convert_to_tensor(target, tf.float32)
    return dna_encoding, seq_feature, target


def data_func_one(feature, target):
    """convert to tensor"""
    feature = tf.convert_to_tensor(feature, tf.float32)
    target = tf.convert_to_tensor(target, tf.float32)
    return feature, target


def predict_data_func(dna_encoding, seq_feature):
    """convert to tensor"""
    dna_encoding = tf.convert_to_tensor(dna_encoding, tf.float32)
    seq_feature = tf.convert_to_tensor(seq_feature, tf.float32)
    return dna_encoding, seq_feature


def predict_data_func_one(feature):
    """convert to tensor"""
    feature = tf.convert_to_tensor(feature, tf.float32)
    return feature


def get_dataset(samples, reference_genome_file, sequencing_data_file, target_sequencing_file=None, window_width=128, extend=40960, data_type='dna+seq', nan=None):
    """
    A fuction to load the feature of each sample to GPU

    Args:
        samples: numpy data frame, shape:[num_sample, 3]
        reference_genome_file: the path of the reference genome date file
        sequence_data_path: the path of the sequencing bigwig file
        target_sequencing_file: the path of the ground truth or label file
        window_width: the genomic signal within the window of length window_width will be represented as a value
        extend: the length extended on both sides of each sample in order to take full advantage of the transformer
        task_type: classificaiton or regression
        data_type: feature types for user supplied data
        nan: replace 'Nan' or 'Inf' in the data with the parameter value

    Return:
        dataset: a dataloader
    """
    ####################################
    # training and evaluation
    ####################################
    # select data loader function
    if target_sequencing_file is not None:
        if data_type == 'dna+seq':
            get_samples_function = get_dna_seq_features
            mac_func = data_func
        elif data_type == 'seq+seq':
            get_samples_function = get_seq_seq_features
            mac_func = data_func
        elif data_type == 'dna':
            get_samples_function = get_dna_features
            mac_func = data_func_one
        elif data_type == 'seq':
            get_samples_function = get_seq_features
            mac_func = data_func_one
        
        sample_func, sample_types, sample_shapes = get_samples_function(np.asarray(samples),
                                                                       reference_genome_file,
                                                                       sequencing_data_file,
                                                                       target_sequencing_file,
                                                                       window_width=window_width,
                                                                       extend=extend,
                                                                       nan=nan,
                                                                       )
    ####################################
    # prediction
    ####################################
    # select data loader function
    else:
        if data_type == 'dna+seq':
            get_samples_function = get_predict_dna_seq
            mac_func = predict_data_func
        elif data_type == 'seq+seq':
            get_samples_function = get_predict_seq_seq
            mac_func = predict_data_func
        elif data_type == 'dna':
            get_samples_function = get_predict_dna
            mac_func = predict_data_func_one
        elif data_type == 'seq':
            get_samples_function = get_predict_seq
            mac_func = predict_data_func_one
        sample_func, sample_types, sample_shapes = get_samples_function(np.asarray(samples),
                                                                    reference_genome_file,
                                                                    sequencing_data_file,
                                                                    extend=extend,
                                                                    nan=nan)

    dataset = tf.data.Dataset.from_generator(sample_func, sample_types, sample_shapes)
    dataset = dataset.map(map_func=mac_func, num_parallel_calls=100)
    return dataset

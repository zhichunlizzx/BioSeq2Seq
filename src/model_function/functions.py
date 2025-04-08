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
from model_function.dataloader import get_dataset
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

        bw_path = os.path.join(out_path, target_list[j] + '.bw')

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
    
    test_dataset = get_dataset(samples, reference_genome_file, sequencing_data_file, data_type=data_type, extend=extend, nan=nan).batch(1)

    # record results
    for j, data in tqdm(enumerate(test_dataset)):
        result = {}
        predicted_tf = predict(data)
        # print(predicted_tf)
        result['chr'] = samples[j][0]
        result['start'] = int(samples[j][1])
        result['end'] = int(samples[j][2])
        result['predicted'] = predicted_tf[0].numpy()
        results[result['chr']].append(result)
        # print(tf.reduce_max(predicted_tf))

    write_down = write_predicted_result(results, out_path, chr_length, target_list, reference_genome_idx, seq_length=seq_length, window_size=window_size)

    os.remove(reference_genome_idx)
        
    return True


def _reduced_shape(shape, axis):
    if axis is None:
        return tf.TensorShape([])
    return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
    """Contains shared code for PearsonR and R2."""

    def __init__(self, reduce_axis=None, name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation (say
            (0, 1). If not specified, it will compute the correlation across the
            whole tensor.
          name: Metric name.
        """
        super(CorrelationStats, self).__init__(name=name)
        self._reduce_axis = reduce_axis
        self._shape = None  # Specified in _initialize.

    def _initialize(self, input_shape):
        # Remaining dimensions after reducing over self._reduce_axis.
        self._shape = _reduced_shape(input_shape, self._reduce_axis)

        weight_kwargs = dict(shape=self._shape, initializer='zeros')
        self._count = self.add_weight(name='count', **weight_kwargs)
        self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
        self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
        self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                                 **weight_kwargs)
        self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
        self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                                 **weight_kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state.

        Args:
          y_true: Multi-dimensional float tensor [batch, ...] containing the ground
            truth values.
          y_pred: float tensor with the same shape as y_true containing predicted
            values.
          sample_weight: 1D tensor aligned with y_true batch dimension specifying
            the weight of individual observations.
        """
        if self._shape is None:
            # Explicit initialization check.
            self._initialize(y_true.shape)
        y_true.shape.assert_is_compatible_with(y_pred.shape)
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        self._product_sum.assign_add(
            tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

        self._true_sum.assign_add(
            tf.reduce_sum(y_true, axis=self._reduce_axis))

        self._true_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

        self._pred_sum.assign_add(
            tf.reduce_sum(y_pred, axis=self._reduce_axis))

        self._pred_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

        self._count.assign_add(
            tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

    def result(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    def reset_states(self):
        if self._shape is not None:
            tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                              for v in self.variables])


class PearsonR(CorrelationStats):
    """Pearson correlation coefficient.

    Computed as:
    ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
    """

    def __init__(self, reduce_axis=(0,), name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation.
          name: Metric name.
        """
        super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                       name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count

        covariance = (self._product_sum
                      - true_mean * self._pred_sum
                      - pred_mean * self._true_sum
                      + self._count * true_mean * pred_mean)

        true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
        pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
        tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
        correlation = covariance / tp_var

        return correlation


class R2(CorrelationStats):
    """R-squared  (fraction of explained variance)."""

    def __init__(self, reduce_axis=None, name='R2'):
        """R-squared metric.

        Args:
            reduce_axis: Specifies over which axis to compute the correlation.
            name: Metric name.
        """
        super(R2, self).__init__(reduce_axis=reduce_axis,
                                 name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        total = self._true_squared_sum - self._count * tf.math.square(true_mean)
        residuals = (self._pred_squared_sum - 2 * self._product_sum
                     + self._true_squared_sum)

        return tf.ones_like(residuals) - residuals / total


class MetricDict:
    def __init__(self, metrics):
        self._metrics = metrics

    def update_state(self, y_true, y_pred):
        for k, metric in self._metrics.items():
            metric.update_state(y_true, y_pred)

    def result(self):
        return {k: metric.result() for k, metric in self._metrics.items()}


def make_length_dict(path):
    '''Record the length of each chromosome'''
    length_dict = {}
    for line in open(path):
        a = line.split()
        length_dict[a[0]] = int(a[2])
    return length_dict


def create_step_function(model, optimizer):
    """Train model and update the model"""
    @tf.function
    def train_step(data_item, target, epoch, optimizer_clip_norm_global=0.2):
        # Forward
        with tf.GradientTape() as tape:
            outputs_tf = model(data_item, is_training=True)
            
            loss = tf.reduce_mean(tf.keras.losses.MSE(target, outputs_tf))

        # backpropagation
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, global_norm = tf.clip_by_global_norm(gradients, 5)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, outputs_tf
    return train_step


def resolution_128_to(target, n = 8):
    target = rearrange(target, 'b (r n) d -> b r n d', n=n)
    target = tf.reduce_mean(target, axis=2)
    return target


def regressive_model_evaluation(valid_samples,
                            reference_genome_file,
                            sequencing_data_file,
                            target_sequencing_file,
                            trained_model,
                            batch_size,
                            window_width,
                            data_type='dna+seq',
                            max_steps=None,
                            extend=40960,
                            nan=None
                            ):
    """
    evaluate the model

    Return:
        metric: correlation
        loss
    """
    @tf.function
    def predict(data_item):
        return trained_model(data_item, is_training=False)
     
    valid_dataset = get_dataset(valid_samples,
                                reference_genome_file,
                                sequencing_data_file,
                                target_sequencing_file,
                                window_width=window_width,
                                data_type=data_type,
                                extend=extend,
                                nan=nan,
                                ).batch(batch_size)

    
    # evaluation
    metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})

    loss = 0
    for i, data_item in tqdm(enumerate(valid_dataset)):
        if max_steps is not None and i > max_steps:
            break

        test_target = data_item[-1]
        outputs_tf = predict(data_item[:2])
        
        loss = tf.reduce_mean(tf.keras.losses.MSE(test_target, outputs_tf))
        metric.update_state(test_target, outputs_tf)

    if max_steps is not None:
        return metric.result()['PearsonR'].numpy(), loss.numpy() / i
    else:
        return metric.result()['PearsonR'].numpy(), loss.numpy() / len(valid_dataset)


def model_train(train_samples,
                validation_samples,
                reference_genome_file,
                sequencing_data_file,
                target_sequencing_file,
                model,
                lr,
                batch_size,
                epoch_num,
                step_per_epoch,
                window_width=128,
                data_type='dna+seq',
                extend=40960,
                nan=0,
                valid_max_steps=100000,
                save_path=None,
                evaluation_epoch_num=1,
                target_list=[],
                lr_attenuation=1.5,
                lr_trans_epoch=1,
                ):
    """ 
    train the model

    Args:
        train_samples: training samples, shape:[num_sample, 3]
        reference_genome_file: the path of the reference genome file
        sequencing_data_file: the path of sequencing files (like ChIP-seq, ATAC-seq,...)
        target_sequencing_file: the path of the ground truth or label file
        model: the model has been build
        lr: learning rate
        batch_size: the amount of data fed to the model per parameter iteration
        epoch_num: number of iterations to train the model using the training set
        step_per_epoch: 
        window_width: resolution of the model. signals within each window_width range will be predicted as a value
        data_type: the data type of the input data of model
        extend: the length extended on both sides of each sample in order to take full advantage of the transformer
        nan: replace Nan or Inf in the data with the parameter value
        lr_attenuation: the multiplier by which the learning rate is reduced after each epoch
        lr_trans_epoch: learning rate will change each lr_trans_epoch

    Return:
        model: trained model
        loss_per_epoch: loss [num_epoch, 1]
    """

    # tensorboard object
    log_dir = os.path.join(save_path, 'log')
    summary_writer = tf.summary.create_file_writer(log_dir)

    learning_rate = tf.Variable(lr, trainable=False, name='learning_rate')
    loss_per_epoch = []
    target_learning_rate = lr
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # training function
    train_step = create_step_function(model, optimizer)

    samples = train_samples

    global_step = 0
    max_correlation = 0

    for epoch in range(epoch_num):
        print('epoch: ', epoch)
        print('lr: ', target_learning_rate)

        # randomly select samples
        t_samples, _ = split_based_num(samples, step_per_epoch)
        train_dataset = get_dataset(t_samples,
                    reference_genome_file,
                    sequencing_data_file,
                    target_sequencing_file,
                    window_width=window_width,
                    data_type=data_type,
                    extend=extend,
                    nan=nan).batch(batch_size).repeat().prefetch(10)
        train_iter = iter(train_dataset)
        
        # training
        epoch_loss = 0.
        for i in tqdm(range(step_per_epoch)):
            global_step += 1
            if global_step > 1:
                learning_rate_frac = tf.math.minimum(
                    1.0, global_step / tf.math.maximum(1.0, step_per_epoch))
                learning_rate.assign(target_learning_rate * learning_rate_frac)
                    
            data_item = next(train_iter)
            input_data = data_item[:-1]

            loss, out = train_step(input_data, data_item[-1], epoch)
            epoch_loss += loss.numpy().astype('float32')

        with summary_writer.as_default():
            tf.summary.scalar("train_loss", np.mean(epoch_loss / step_per_epoch), step=epoch)
        
        loss_per_epoch.append(np.mean(epoch_loss / step_per_epoch))

        if epoch % evaluation_epoch_num == 0:
            evaluation_results, evaluation_loss = regressive_model_evaluation(validation_samples,
                                                                reference_genome_file,
                                                                sequencing_data_file,
                                                                target_sequencing_file,
                                                                model,
                                                                batch_size,
                                                                window_width, 
                                                                data_type=data_type,
                                                                max_steps=valid_max_steps,
                                                                extend=extend,
                                                                nan=nan
                                                            )
            print(evaluation_results)
            with summary_writer.as_default():
                tf.summary.scalar("valid_loss", evaluation_loss, step=epoch)
                for item in range(len(target_list)):
                    tf.summary.scalar(target_list[item], evaluation_results[item], step=epoch)

        # Adjusted learning rate
        if epoch % lr_trans_epoch == 0 and epoch > 0:
            target_learning_rate = target_learning_rate / lr_attenuation

        # save model
        if np.mean(evaluation_results) > max_correlation:
            model.save_weights(os.path.join(save_path, 'best_model/model.ckpt'))
            max_correlation = np.mean(evaluation_results)

    # save model of the last epoch
    model.save_weights(os.path.join(save_path, 'last_model/model.ckpt'))
                                              
    return model, loss_per_epoch


def model_evaluation(valid_samples,
                reference_genome_file,
                sequencing_data_file,
                target_sequencing_file,
                trained_model,
                batch_size,
                window_width, 
                data_type='dna+seq',
                max_steps=None,
                extend=40960,
                nan=None
                ):

    """ 
    evaluate the model

    Args:
        valid_samples: validation samples, shape:[num_sample, 3]
        reference_genome_file: the path of the reference genome file
        sequencing_data_file: the path of sequencing files (like ChIP-seq, ATAC-seq,...)
        model: the model has been build
        lr: learning rate
        batch_size: the amount of data fed to the model per parameter iteration
        epoch_num: number of iterations to train the model using the training set
        window_width: resolution of the model. signals within each window_width range will be predicted as a value
        data_type: the data type of the input data of model
        extend: the length extended on both sides of each sample in order to take full advantage of the transformer
        nan: replace Nan or Inf in the data with the parameter value
    
    Return:
        evaluation_results: pearsonr correlation or accuracy
        evaluation_loss: loss [num_epoch, 1]
    """

    evaluation_function = regressive_model_evaluation
    evaluation_results, evaluation_loss = evaluation_function(valid_samples,
                                                            reference_genome_file,
                                                            sequencing_data_file,
                                                            target_sequencing_file,
                                                            trained_model,
                                                            batch_size,
                                                            window_width, 
                                                            data_type=data_type,
                                                            max_steps=max_steps,
                                                            extend=extend,
                                                            nan=nan
                                                            )
                                                            
    return evaluation_results, evaluation_loss


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

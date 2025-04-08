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
from sample.candidate_region import select_candidate_regions
from sample.positive_samples import get_samples_based_peak
from sample.negative_samples import get_samples_except_peak
from sample.wholegenome_samples import get_predicted_samples
from model_function.functions import load_chromosomes, bw_2_chromosome_size, model_train, model_evaluation, predicted_to_bigwig, split_based_chr, split_based_percent, split_based_num, check_if_out_of_bounds
import numpy as np
import json
import pysam
from model_function.get_feature import fetch_dna
from HistoneModification.model_histonemodification_one_input import HMModel as Regression_Model_one
from HistoneModification.model_histonemodification import HMModel as Regression_Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class SamplePreprocess(object):
    def __init__(self,
                reference_genome_file=None,
                sequencing_data_file=None,
                blacklist_file=None,
                except_bed_file:list=[],
                include_chr:list=[],
                except_chr:list=[]
                ):
        """
        Args:
            reference_genome_file: the path of the reference genome date file
            sequencing_data_file: the path of the sequencing bigwig file
            blacklist_file: blacklist file
            except_bed_file: regions that need to be removed except blacklist
            include_chr: chromosomal data needed for training or prediction
            except_chr: chromosome data not needed for training or prediction
        """
        super(SamplePreprocess, self).__init__()

        # whether reference_genome_file and sequencing_data_file exist
        if reference_genome_file is not None:
            if not os.path.exists(reference_genome_file):
                raise Exception('Error: File %s does not exist' % reference_genome_file)
            chrom_size = load_chromosomes(reference_genome_file)
        elif sequencing_data_file is not None:
            for seq_file_group in sequencing_data_file[0]:
                for seq_file in seq_file_group:
                    if not os.path.exists(seq_file):
                        raise Exception('Error: File %s does not exist' % sequencing_data_file)
            chrom_size = bw_2_chromosome_size(sequencing_data_file[0][0][0])

        # reference_genome_file and sequencing_data_file must have one
        if reference_genome_file is None and sequencing_data_file is None:
            raise Exception('Error: reference_genome_file or sequencing_data_file must be provided')

        # for predicting
        self.whole_genome_size = chrom_size

        # for training and validation
        self.train_chrom_size = {}
        if len(include_chr) > 0:
            for chr in include_chr:
                self.train_chrom_size[chr] = chrom_size[chr]
        elif len(except_chr) > 0:
            for chr in chrom_size:
                if not(chr in except_chr):
                    self.train_chrom_size[chr] = chrom_size[chr]
        else:
            self.train_chrom_size = chrom_size

        self.include_chr = include_chr
        self.except_chr = except_chr
        self.blacklist_file = blacklist_file
        self.except_bed_file = except_bed_file

        self.reference_genome_file = reference_genome_file
        self.sequencing_data_file = sequencing_data_file


    def get_train_chrom_size(self):
        return self.train_chrom_size
    
    
    def get_whole_genome_size(self):
        return self.whole_genome_size
    

    def set_include_chr(self, include_chr):
        self.include_chr = include_chr
        for chr in include_chr:
            self.train_chrom_size[chr] = self.whole_genome_size[chr]
  

    def get_include_chr(self):
        return self.include_chr


    def set_except_chr(self, except_chr):
        self.except_chr = except_chr
        for chr in self.whole_genome_size:
            if not(chr in except_chr):
                self.train_chrom_size[chr] = self.whole_genome_size[chr]
  

    def get_except_chr(self):
        return self.except_chr


    def get_candidate_regions(self):
        """ remove regions of no interest from genome-wide """
        self.candidate_regions = select_candidate_regions(
                                                        self.train_chrom_size,
                                                        self.blacklist_file,
                                                        self.except_bed_file,
                                                        self.include_chr
                                                        )
        return self.candidate_regions
    
    
    def get_positive_samples(self,
                            seq_length,
                            overlap,
                            input_data_peak_paths=[],
                            output_data_peak_paths=[]
                            ):
        """ get positive samples based peak regions """

        if self.candidate_regions is None:
            raise Exception('Error: please get the candidate area first')
        
        self.positive_samples = get_samples_based_peak(
                                                self.candidate_regions,
                                                input_data_peak_paths,
                                                output_data_peak_paths,
                                                seq_length=seq_length,
                                                )
        return self.positive_samples
    

    def get_negative_samples(self,
                            seq_length,
                            overlap,
                            input_data_peak_paths:list=[],
                            output_data_peak_paths:list=[]):
        """ get negative samples from the complement of peak """

        if self.candidate_regions is None:
            raise Exception('Error: please get the candidate area first')
        self.negative_samples = get_samples_except_peak(
                                                    self.candidate_regions,
                                                    input_data_peak_paths,
                                                    output_data_peak_paths,
                                                    seq_length=seq_length,
                                                    overlap=overlap
                                                    )
        return self.negative_samples
        

    def save_samples(self, samples, sample_path):
        with open(sample_path, 'w') as w_obj:
            for sample in samples:
                sample = [str(item) for item in sample]
                w_obj.write('\t'.join(sample) + '\n')


    def load_samples(self, sample_path):
        self.samples = np.loadtxt(sample_path, dtype=str, delimiter='\t')
        return self.samples
    

    def get_samples(self,
                    seq_length:int=114688,
                    overlap:int=81920,
                    peak:bool=True,
                    input_data_peak_paths:list=[],
                    output_data_peak_paths:list=[]
                    ):

        """
        get the positive and negative samples at once

        Args:
            seq_length: genomic length covered by a sample

            overlap: overlap greater than 0 means that positive examples with overlap will be generated

            peak: bool, whether to use peak as the positive samples

            input_data_peak_paths: the peak bed file of the sequencing data for inferring other types of geneomic information

            output_data_peak_paths: the peak bed file of the sequencing data is used as the ground truth or label
            
        Return:
            self.samples: samples (num_sample, 3])
        """
        
        self.candidate_regions = self.get_candidate_regions()

        # whether to artificially provide data in information-rich genome regions
        if peak:
            if len(input_data_peak_paths) == 0 and len(output_data_peak_paths) == 0:
                raise Exception('Please provide at least one input_data_peak_paths or output_data_peak_paths')
            else:
                self.positive_samples = self.get_positive_samples(seq_length,
                                                                overlap,
                                                                input_data_peak_paths,
                                                                output_data_peak_paths
                                                                )
                self.negative_samples = self.get_negative_samples(seq_length,
                                                                overlap,
                                                                input_data_peak_paths,
                                                                output_data_peak_paths
                                                                )

                self.samples = np.concatenate((self.positive_samples, self.negative_samples))
        else:
            self.samples = self.get_negative_samples(seq_length, overlap)

        np.random.shuffle(self.samples)

        return self.samples


    def data_cleansing(self, samples):
        raw_samples = samples
        clean_data = []
        if self.reference_genome_file is not None:
            try:
                fasta_open = pysam.Fastafile(self.reference_genome_file)
            except:
                raise Exception('Error: %s is not the correct reference genome file' % self.reference_genome_file)

            for sample in raw_samples:
                dna_code = np.asarray(fetch_dna(fasta_open, sample[0], int(sample[1]), int(sample[2])))
                if np.all(dna_code=='N'):
                    continue
                clean_data.append(sample)

            return np.asarray(clean_data)
        else:
            return samples
            

    def split_samples(self, samples, split_parameter):
        """
        split samples into two set
        
        Args:
            samples: samples to be splited
            split_parameter: chr list, split number, or split proportion

        Return:
            splited_samples: samples selected by 'split_parameter'
            remaining_samples: the remain samples after selecting 'splited samples'
        """

        # determine the type of 'split_parameter'
        if type(split_parameter) == list:
            divide_function = split_based_chr
        elif type(split_parameter) == int and split_parameter >= 0:
            divide_function = split_based_num
        elif type(split_parameter) == float and split_parameter >= 0 and split_parameter <= 1:
            divide_function = split_based_percent
        else:
            raise Exception("Error: please provide the correct divide_parameter(str, float(0=<x<=1), int(x>=0))")

        # split samples to two sets
        splited_samples, remaining_samples = divide_function(samples, split_parameter)

        return splited_samples, remaining_samples
    

    def get_evaluation_samples(self,
                        seq_length:int=114688,
                        include_chr:list=['chr22'],
                        blacklist_file=None,
                        start_posi=None,
                        ):
        """ get the test samples, there is no overlap between any two samples. """
        self.predicted_samples = get_predicted_samples(self.whole_genome_size,
                                                       include_chr,
                                                       seq_length,
                                                       blacklist_file,
                                                       start_posi=start_posi,
                                                       )
        
        return self.predicted_samples
    

class BioSeq2Seq(object):
    """for training, evaluation and prediction"""

    def __init__(self,
                reference_genome_file=None,
                sequencing_data_file: list=[],
                target_sequencing_file: list=[],):
        super(BioSeq2Seq, self).__init__()
        """
        Args:
            reference_genome_file: the path of the reference genome date file
            sequencing_data_file: the path of the sequencing bigwig file
            target_sequencing_file: the path of the ground truth or label file
        """

        # check input data
        # whether reference_genome_file and sequencing_data_file exist
        if reference_genome_file is not None:
            if not os.path.exists(reference_genome_file):
                raise Exception('Error: File %s does not exist' % reference_genome_file)
            chrom_size = load_chromosomes(reference_genome_file)
        elif sequencing_data_file is not None:
            if len(sequencing_data_file) > 0:
                for input_seq_files in sequencing_data_file:
                    single_or_double = len(input_seq_files[0])
                    for seq_files in input_seq_files:
                        if len(seq_files) != single_or_double:
                            raise Exception('Error: the same sequencing data should be all single-chain data or double-chain data')
                        for file in seq_files:
                            if not os.path.exists(file):
                                raise Exception('Error: File %s does not exist' % file)
            chrom_size = bw_2_chromosome_size(sequencing_data_file[0][0][0])
        else:
            raise Exception('Error: please provide at least one sequencing_data_file or reference_genome_file')
        
        # check output data
        # whether every file in list target_sequencing_file exist
        for file in target_sequencing_file:
            if not os.path.exists(file):
                raise Exception('Error: %s does not exist' % file)

        self.reference_genome_file = reference_genome_file
        self.sequencing_data_file = sequencing_data_file
        self.target_sequencing_file = target_sequencing_file

        self.model = None

        # training parameters
        self.model_config = {}
        self.model_config['target_list'] = None
        self.model_config['chrom_size'] = chrom_size
        self.model_config['data_type'] = None
        self.model_config['seq_length'] = None
        self.model_config['window_width'] = None
        self.model_config['extend'] = None
        self.model_config['channels'] = None
        self.model_config['num_heads'] = None
        self.model_config['num_transformer_layers'] = None
        

    def identify_data_type(self, reference_genome_file, sequencing_data_file):
        """ identify the type of data entered by the user """
        if reference_genome_file is None:
            if len(sequencing_data_file) == 1:
                data_type = 'seq'
            elif len(sequencing_data_file) == 2:
                data_type = 'seq+seq'
            else:
                raise Exception('Error: only a maximum of two inputs are supported')

        else:
            if len(sequencing_data_file) == 0:
                data_type = 'dna'
            elif len(sequencing_data_file) == 1:
                data_type = 'dna+seq'
            else:
                raise Exception('Error: only a maximum of two inputs are supported')
        
        return data_type
    

    def build_model(self,
                    target_list:list,
                    seq_length:int=114688,
                    window_width:int=128,
                    num_heads:int=8,
                    channels:int=768,
                    num_transformer_layers:int=11,
                    extend:int=40960,
                    nan=None,
                    init_lr=0.001,
                    task='TFBS',
                    ):

        """
        build a BioSeq2Seq model based parameters provided by users

        Args:
            target_list: the model outputs the biological name corresponding to each track
            seq_length: genomic length covered by a sample
            window_width: the genomic signal within the window of length window_width will be represented as a value
            num_heads: the number of head of MuitiHeadAttention layers
            channels: channel of model
            num_tramsfprmer_layers: the number of transformer layers
            extend: the length extended on both sides of each sample in order to take full advantage of the transformer
            nan: replace 'Nan' or 'Inf' in the data with the parameter value
            init_lr: initial learning rate

        Return:
            self.model: the BioSeq2Seq model
        """
        
        if seq_length % window_width > 0:
            raise Exception('seq_length must be divisible by window_width')
        if extend % window_width > 0:
            raise Exception('extend must be divisible by window_width')
        if len(self.target_sequencing_file) > 0:
            if len(self.target_sequencing_file) != len(target_list):
                raise Exception('num of target_sequencing_file must be equal to the length of target_list')
        else:
            raise Exception('must provide at least one target sequencing file')
            
        self.seq_length = seq_length
        
        self.model_config['window_width'] = window_width
        self.model_config['extend'] = extend
        self.model_config['target_list'] = target_list
        self.model_config['target_length'] = seq_length // window_width
        self.model_config['data_type'] = self.identify_data_type(self.reference_genome_file, self.sequencing_data_file)

        self.model_config['channels'] = channels
        self.model_config['num_heads'] = num_heads
        self.model_config['num_transformer_layers'] = num_transformer_layers
        self.model_config['nan'] = nan
        self.model_config['init_lr'] = init_lr

        # if task == 'HM':
        #     from HistoneModification.model_histonemodification_one_input import HMModel as Regression_Model_one
        #     from HistoneModification.model_histonemodification import HMModel as Regression_Model
        # elif task == 'FE':
        #     from FunctionalElements.model_FE_one_inputs import FEModel as Regression_Model_one
        #     from FunctionalElements.model_FE import FEModel as Regression_Model
        # elif task == 'GE':
        #     from GeneExpression.model_geneExpression_one_input import GEModel as Regression_Model_one
        #     from GeneExpression.model_geneExpression import GEModel as Regression_Model
        # elif task == 'TF':
        #     from TFBS.model_TFBS_one_input import TFModel as Regression_Model_one
        #     from TFBS.model_TFBS import TFModel as Regression_Model

        # select model
        if self.model_config['data_type'] == 'seq' or self.model_config['data_type'] == 'dna':
            model = Regression_Model_one
        elif self.model_config['data_type'] == 'seq+seq' or self.model_config['data_type'] == 'dna+seq':
            model = Regression_Model


        # build model
        self.model = model(
            channels=self.model_config['channels'],
            num_heads=self.model_config['num_heads'],
            num_transformer_layers=self.model_config['num_transformer_layers'],
            pooling_type='max',
            output_channels=len(self.model_config['target_list']),
            target_length=self.model_config['target_length']
        )

        return self.model


    def evaluation(self,
                   validation_samples,
                   batch_size=1,
                   evaluation_step=1000000):
        """ 
        evaluate the performence of the model
        
        Return:
            evaluation_results: pearson correlation
            evaluation_loss: evaluation loss
        """
        
        if self.model is None:
            raise Exception('Error: please build model first')

        check = check_if_out_of_bounds(validation_samples, self.model_config['chrom_size'])
        if check is not None:
            raise Exception('Some valid samples in %s out of bounds' % check)

        evaluation_results, evaluation_loss = model_evaluation(validation_samples,
                                                            self.reference_genome_file,
                                                            self.sequencing_data_file,
                                                            self.target_sequencing_file,
                                                            self.model,
                                                            batch_size,
                                                            window_width=self.model_config['window_width'],
                                                            data_type=self.model_config['data_type'],
                                                            max_steps=evaluation_step,
                                                            extend=self.model_config['extend'],
                                                            nan=self.model_config['nan']
                                                            )
        
        return evaluation_results, evaluation_loss


    def train(self,
            train_samples,
            validation_samples,
            batch_size=1,
            epoch_num=100,
            evaluation_epoch_num=1,
            step_per_epoch=5000,
            valid_max_steps=100000,
            save_path=None,
            lr_attenuation=10,
            lr_trans_epoch=10,
            
            ):
        """
        train and evaluation

        Args:
            train_samples: numpy data frame, shape:[num of training samples, 3]
            validation_samples: numpy data frame, shape:[num of validation sample, 3]
            batch_size: batch size
            epoch_num: iteractions
            evaluation_epoch_num: evaluation the model  every 'evaluation_epoch_num' epoch
            step_per_epoch: number of samples trained per epoch
            valid_max_steps: number of samples used to evaluate
            save_path: save path of model
            lr_attenuation, lr_trans_epoch: the multiplier by which the learning rate is reduced for each 'lr_trans_epoch' epoch

        Return:
            model: trained model
            train_loss: loss of each epoch
            evaluation_results: pearson correlation of each target
            evaluation_loss: loss of evluation
        """
        
        if self.model is None:
            raise Exception('Error: please build model first')
        
        if epoch_num < evaluation_epoch_num:
            raise Exception('Error: epoch_num must greater than evaluation_epoch_num')

        if epoch_num % evaluation_epoch_num > 0:
            raise Exception('Error: epoch_num must be divisible by evaluation_epoch_num.')

        if step_per_epoch > len(train_samples):
            raise Exception('Error: number of train_samples must greater than step_per_epoch')

        check = check_if_out_of_bounds(train_samples, self.model_config['chrom_size'])
        if check is not None:
            print('Some training samples in %s is out of bounds' % check)
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        lr = self.model_config['init_lr']

        model = self.model
        
        self.save_model(os.path.join(save_path, 'best_model'))
        model, train_loss= model_train(train_samples,
                                    validation_samples,
                                    self.reference_genome_file,
                                    self.sequencing_data_file,
                                    self.target_sequencing_file,
                                    model,
                                    lr,
                                    batch_size,
                                    epoch_num=epoch_num,
                                    evaluation_epoch_num=evaluation_epoch_num,
                                    step_per_epoch=step_per_epoch,
                                    window_width=self.model_config['window_width'],
                                    data_type = self.model_config['data_type'],
                                    extend=self.model_config['extend'],
                                    nan=self.model_config['nan'],
                                    valid_max_steps=valid_max_steps,
                                    save_path=save_path,
                                    target_list=self.model_config['target_list'],
                                    lr_attenuation=lr_attenuation,
                                    lr_trans_epoch=lr_trans_epoch,
                                    )
        self.save_model(os.path.join(save_path, 'best_model'))
        return model, train_loss
    

    def predict(self,
                predicted_samples,
                out_path: str=None,
                reference_genome_file: str=None,
                sequencing_data_file:list=[]
                ):
        """output the predicted result for whole genome based the fitting model"""

        # check samples
        check = check_if_out_of_bounds(predicted_samples, self.model_config['chrom_size'])
        if check is not None:
            print('Some predicted samples in %s out of bounds' % check)
        
        data_type = self.identify_data_type(reference_genome_file, sequencing_data_file)
        if data_type != self.model_config['data_type']:
            raise Exception('Error: please provide the correct data %s' % self.model_config['data_type'])

        # predict and write the results to bigwig files
        predicted_state = predicted_to_bigwig(self.model,
                                            predicted_samples,
                                            reference_genome_file,
                                            sequencing_data_file,
                                            self.model_config['target_list'],
                                            self.model_config['chrom_size'],
                                            out_path,
                                            data_type,
                                            extend=self.model_config['extend'],
                                            nan=self.model_config['nan'],
                                            seq_length=self.seq_length,
                                            window_size=self.model_config['window_width'],
                                            )
        if predicted_state == True:        
            print('predicted has been completed')


    def save_model(self, path):
        """save model"""

        if self.model is None:
            raise Exception('Error: need to build_model before saving the model')

        model_outpath = os.path.join(path, 'model.ckpt')
        self.model.save_weights(model_outpath)
        self.save_config(path)


    def save_config(self, path):
        """save parameters of the model"""

        config_outpath = os.path.join(path, 'model.config')

        with open(config_outpath, 'w') as stats_json_out:
            json.dump(self.model_config, stats_json_out, indent=4)


    def get_model_config(self):
        """view parameters"""
        return self.model_config

     
    def load_model(self, path):
        """load model and parameters"""
        config_path = os.path.join(path, 'model.config')
        model_path = os.path.join(path, 'model.ckpt')

        # load parameters
        with open(config_path, 'r') as r_obj:
            self.model_config = json.load(r_obj)

        # select model
        if self.model_config['data_type'] == 'seq' or self.model_config['data_type'] == 'dna':
            model = Regression_Model_one
        elif self.model_config['data_type'] == 'seq+seq' or self.model_config['data_type'] == 'dna+seq':
            model = Regression_Model

        # build model
        self.model = model(
            channels=self.model_config['channels'],
            num_heads=self.model_config['num_heads'],
            num_transformer_layers=self.model_config['num_transformer_layers'],
            pooling_type='max',
            output_channels=len(self.model_config['target_list']),
            target_length=self.model_config['target_length']
        )

        # load model
        self.model.load_weights(model_path)

        return self.model
    
    def load_weights(self, path):
        """load model and parameters"""
        model_path = os.path.join(path, 'model.ckpt')

        # load model
        self.model.load_weights(model_path)

        return self.model
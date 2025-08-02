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
from candidate_region import select_candidate_regions
from positive_samples import get_samples_based_peak
from negative_samples import get_samples_except_peak
from wholegenome_samples import get_predicted_samples
from functions import load_chromosomes, bw_2_chromosome_size,  split_based_chr, split_based_percent, split_based_num, check_if_out_of_bounds
import numpy as np
import json
import pysam
from get_feature import fetch_dna



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
    



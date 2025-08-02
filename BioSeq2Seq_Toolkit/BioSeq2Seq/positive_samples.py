#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import os
import subprocess
import numpy as np
import shutil


def get_overlap(histone_section, dREG_section):
    """compute overlap beteen two regions: histone_section and dREG_section"""
    h_start, h_end = int(histone_section['start']), int(histone_section['end'])
    d_start, d_end = int(dREG_section['start']), int(dREG_section['end'])
    if h_start > d_end or h_end < d_start:
        return 0
    if d_start > h_start and d_end < h_end:
        return d_end - d_start + 1
    if d_start >= h_start:
        return h_end - d_start + 1
    if h_start >= d_start:
        return d_end - h_start + 1


def add_seq_num(histone_dict, dREG_dict, dREG_label):
    """select which roseq data to use"""
    for chr in histone_dict:
        histone_sections = histone_dict[chr]
        dREG_sections = dREG_dict[chr]
        for i in range(len(histone_sections)):
            for dREG_section in dREG_sections:
                overlap = get_overlap(histone_sections[i], dREG_section)
                if overlap > histone_sections[i]['overlap']:
                    histone_sections[i]['seq_num'] = dREG_label
                    histone_sections[i]['overlap'] = overlap
        histone_dict[chr] = histone_sections
    return histone_dict


def add_key_values(dict_data, key_name, value):
    """add new key-value pairs to dict_data"""
    for chr in dict_data:
        chr_dicts = dict_data[chr]
        chr_dicts = [{**chr_dict, key_name: value} for chr_dict in chr_dicts]

        dict_data[chr] = chr_dicts
    return dict_data


def convert_dict_2_bed_dataframe(dict_data):
    """ 
    Convert the data recorded in the dictionary into the data format of the bed file

    Args:
        dict_data: {chr:[{start:xx, end:xx}]}

    Return:
        bed_df_data: [[chr, start, end], ...], dimension: [num_samples, 4]
    """
    bed_df_data = []
    for chr in dict_data:
        for region in dict_data[chr]:
            bed_df_data.append([chr, region['start'], region['end'], region['seq_num']])
    return bed_df_data


def read_bed_based_candidate_chromosome(candidate_regions, path):
    """ 
    Read the content in the bed file and store it as a dict

    Args:
        candidate_regions: dict:{chr:[(start0, end0), (start1, end1)], ...}
        path: path of bed file

    Return:
        section_dict: {chr:[{start:xx, end:xx}]}
    """
    with open(path, 'r') as r_obj:
        sections = r_obj.readlines()
    sections = [section.split() for section in sections]

    section_dict = {}

    for chr in candidate_regions:
        section_dict[chr] = []

    for section in sections:
        try:
            section_dict[section[0]].append({'start': int(section[1]), 'end': int(section[2]), 'seq_num': int(section[3])})
        except:
            try:
                section_dict[section[0]].append({'start': int(section[1]), 'end': int(section[2])})
            except:
                pass
    return section_dict


def sample_2_seq_length(section_dict, seq_length, overlap=False):
    """ 
    Select the  genome sample with a length of 114688bp, and combine peaks in the range of 114688 into one sample

    Args:
        section_dict: {chr:[{start:xx, end:xx}]}
        seq_length: length of the sample

    Return:
        section_dict_seq_length: {chr:[{start:xx, end:xx}]}
    """

    section_dict_seq_length = {}

    chr_list = list(section_dict.keys())

    for chr in chr_list:
        sections = section_dict[chr]
        if overlap:
            sections_chr = []
            if len(sections) == 0:
                continue
            sections_chr.append(sections[0])
            for i in range(1, len(sections)):
                # merge peaks within a seq_length range
                if (int(sections[i]['end']) - int(sections_chr[-1]['start'])) < seq_length:
                    sections_chr[-1]['end'] = sections[i]['end']
                else:
                    sections_chr.append(sections[i])
        else:
            sections_chr = sections

        # extend to seq_length
        half_seq_length = seq_length // 2
        sections_chr_seq_length = []
        for section in sections_chr:
            mid = int((int(section['start']) + int(section['end'])) / 2)
            if mid - half_seq_length < 0:
                sections_chr_seq_length.append({'start': 0, 'end': seq_length})
            else:
                sections_chr_seq_length.append({'start': mid - half_seq_length, 'end': mid + half_seq_length})

        section_dict_seq_length[chr] = sections_chr_seq_length

    return section_dict_seq_length


def select_candidate_samples(candidate_regions, candidate_regions_saved_path, unscreened_samples_path, seq_length):
    """ 
    Filter out the samples in the candidate region

    Args:
        candidate_regions: candidate regions  dict:{chr:[(start0, end0), (start1, end1)], ...}
        candidate_regions_saved_path: the path of the bed file of candidate regions
        unscreened_samples_path: the out path of result
        seq_length: length of the sample

    Return:
        candidate_samples: dict:{chr:[(start0, end0), (start1, end1)], ...}
    """
    
    # intersect
    out_candidate_sample_path = os.path.join(os.path.dirname(__file__), 'temporary/candidate_samples.bed')
    cmd = 'bedtools intersect -a ' + unscreened_samples_path + ' -b ' + candidate_regions_saved_path + ' > ' + out_candidate_sample_path
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

    # select samples in candidate regions
    intersect_samples = read_bed_based_candidate_chromosome(candidate_regions, out_candidate_sample_path)

    candidate_samples = {}
    for chr in intersect_samples:
        candidate_samples[chr] = []
        if len(intersect_samples[chr]) > 0:
            for dic in intersect_samples[chr]:
                if int(dic['end']) - int(dic['start']) == seq_length:
                    candidate_samples[chr].append(dic)
    
    return candidate_samples


def write_bed(bed_data_frame, outpath):
    """ 
    Write data to bed file

    Args:
        bed_data_frame : list or dict
        outpath: outpath

    Return:
        None
    """
    try:
        if os.path.isfile(outpath):
            os.remove(outpath)
        with open(outpath, 'w') as w_obj:
            for chr in bed_data_frame:
                if type(bed_data_frame[chr][0]) == dict:
                    for region in bed_data_frame[chr]:
                        w_obj.write(chr + '\t' + str(region['start']) + '\t' + str(region['end']) + '\n')
                else:
                    for region in bed_data_frame[chr]:
                        w_obj.write(chr + '\t' + str(region[0]) + '\t' + str(region[1]) + '\n')                        
        return True
    except:
        return False


def get_samples_without_merge(candidate_regions, input_data_peak_paths, output_data_peak_paths, positive_samples_outpath=os.path.dirname(__file__), seq_length=114688):
    """ 
    Define samples according to the peak of the input sequencing data. 

    Args:
        candidate_regions: candidate regions  dict:{chr:[(start0, end0), (start1, end1)], ...}
        input_data_peak_paths: path of peak of input sequencing data
        output_data_peak_paths: path of peak of output sequencing data
        positive_samples_outpath: outpath of positive samples
        seq_length: length of the sample

    Return:
        None
    """
    temporary_path = os.path.join(os.path.dirname(__file__), 'temporary')
    candidate_regions_saved_path = os.path.join(temporary_path, 'candidate.bed')
    
    # input peak sample
    for i, peak_file_path in enumerate(input_data_peak_paths):
        section_dict = read_bed_based_candidate_chromosome(candidate_regions, peak_file_path)
    
        section_dict = sample_2_seq_length(section_dict, seq_length, overlap=True)

        # write sections to bed file
        section_out_path = os.path.join(os.path.dirname(__file__), 'temporary/sections.bed')
        if os.path.isfile(section_out_path):
            os.remove(section_out_path)
        section_dict_write = write_bed(section_dict, section_out_path)
        
        # select samples in candidate regions
        candidate_samples = select_candidate_samples(candidate_regions, candidate_regions_saved_path, section_out_path, seq_length)
        
        candidate_samples = add_key_values(candidate_samples, 'seq_num', i)
        
        # write positive samples based input sequecing data peak to bed file 
        with open(positive_samples_outpath, 'a') as w_obj:
            for chr in candidate_samples:
                data = candidate_samples[chr]
                for sec in data:
                    w_obj.write(chr + '\t' + str(sec['start']) + '\t' + str(sec['end']) + '\t' + str(sec['seq_num'])  + '\n')

    # target peak sample
    for path in output_data_peak_paths:
        for peak_file in os.listdir(path):
            # print(peak_file)
            peak_file_path = os.path.join(path, peak_file)
            section_dict = read_bed_based_candidate_chromosome(candidate_regions, peak_file_path)

            section_dict = sample_2_seq_length(section_dict, seq_length, overlap=True)

            # write sections to bed file
            section_out_path = os.path.join(os.path.dirname(__file__), 'temporary/sections.bed')
            if os.path.isfile(section_out_path):
                os.remove(section_out_path)
            section_dict_write = write_bed(section_dict, section_out_path)
        
            # select based in candidate regions
            candidate_samples = select_candidate_samples(candidate_regions, candidate_regions_saved_path, section_out_path, seq_length)
            
            # add item seq_num and overlap for selecting seq_num to use
            candidate_samples = add_key_values(candidate_samples, 'seq_num', 0)
            candidate_samples = add_key_values(candidate_samples, 'overlap', 0)

            # determin which seq_num (roseq file) to use
            for i, input_peak_path in enumerate(input_data_peak_paths):
                input_peak_dict = read_bed_based_candidate_chromosome(candidate_regions, input_peak_path)
                candidate_samples = add_seq_num(candidate_samples, input_peak_dict, i)
            
            # write positive samples based input sequecing data peak to bed file
            with open(positive_samples_outpath, 'a') as w_obj:
                for chr in candidate_samples:
                    data = candidate_samples[chr]
                    for sec in data:
                        w_obj.write(chr + '\t' + str(sec['start']) + '\t' + str(sec['end']) + '\t' + str(sec['seq_num']) + '\n')

    return True

def get_samples_based_peak(candidate_regions,
                        input_data_peak_paths=[],
                        output_data_peak_paths=[],
                        seq_length=114688,
                        overlap=0
                        ):
    """ 
    Define samples according to the peak of the target sequencing data

    Args:
        candidate_regions: candidate regions  dict:{chr:[(start0, end0), (start1, end1)], ...}
        input_data_peak_paths: path of peak of input sequencing data
        output_data_peak_paths: path of peak of target sequencing data
        seq_length: length of the sample

    Return:
        candidate_samples: list[num_samples, 3]
    """
    if overlap > 0:
        overlap = True
    else:
        overlap = False

    # creat a temporary folder
    temporary_path = os.path.join(os.path.dirname(__file__), 'temporary')
    if not os.path.isdir(temporary_path):
        os.mkdir(temporary_path)
    
    positive_samples_outpath = os.path.join(temporary_path, 'positives.bed')
    # If there are remnants from the previous run, delete
    if os.path.exists(positive_samples_outpath):
        os.remove(positive_samples_outpath)
    
    candidate_write = write_bed(candidate_regions, os.path.join(temporary_path, 'candidate.bed'))

    # Adjacent peaks in each file are classified into one sample,
    # and there are overlaps between samples generated by different peak files.
    peak_do = get_samples_without_merge(candidate_regions, input_data_peak_paths, output_data_peak_paths, positive_samples_outpath, seq_length)
    
        
    # read bed file based the chromosome in candidate regions
    candidate_samples = read_bed_based_candidate_chromosome(candidate_regions, positive_samples_outpath)

    # covert the sample from dict to (chr, stat, end)
    candidate_samples = convert_dict_2_bed_dataframe(candidate_samples)

    # remove temporary files
    shutil.rmtree(temporary_path)

    return np.asarray(candidate_samples)
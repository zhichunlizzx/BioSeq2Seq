#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import random
import os 
import numpy as np
import sys
import collections
import heapq
from sample.positive_samples import select_candidate_samples, convert_dict_2_bed_dataframe, write_bed, add_key_values
import subprocess
import shutil

Contig = collections.namedtuple('Contig', ['chr', 'start', 'end'])


def remove_peak_regions(chrom_segments, peak_regions):
    """ Remove the genomic interval recorded in peak_regions

    Args:
      chrom_segments: the range of each chromosome, (start,end)
      peak_regions: the file of the interval that needs to be removed

    Returns:
      chrom_segments: same, with segments broken by the assembly gaps.
    """

    chrom_events = {} 
    # add known segments
    for chrom in chrom_segments:
        if len(chrom_segments[chrom]) > 1:
            raise Exception(
            "I've made a terrible mistake...regarding the length of chrom_segments[%s]"
            % chrom,
            file=sys.stderr)

        cstart, cend = chrom_segments[chrom][0]
        chrom_events.setdefault(chrom, []).append((cstart, 'Cstart'))
        chrom_events[chrom].append((cend, 'cend'))

    # add gaps
    for line in peak_regions:
        chrom = line[0]
        gstart = int(line[1])
        gend = int(line[2])

        # consider only if its in our genome
        if chrom in chrom_events:
            chrom_events[chrom].append((gstart, 'gstart'))
            chrom_events[chrom].append((gend, 'Gend'))
 
    # Error correction, judging whether the divided intervals overlap
    for chrom in chrom_events:
        # sort
        chrom_events[chrom].sort()

        # read out segments
        chrom_segments[chrom] = []

        for i in range(len(chrom_events[chrom]) - 1):
            pos1, event1 = chrom_events[chrom][i]
            pos2, event2 = chrom_events[chrom][i + 1]

            event1 = event1.lower()
            event2 = event2.lower()

            shipit = False
            if event1 == 'cstart' and event2 == 'cend':
                shipit = True
            elif event1 == 'cstart' and event2 == 'gstart':
                shipit = True
            elif event1 == 'gend' and event2 == 'gstart':
                shipit = True
            elif event1 == 'gend' and event2 == 'cend':
                shipit = True
            elif event1 == 'gstart' and event2 == 'gend':
                pass
            else:
                print(
                    "I'm confused by this event ordering: %s - %s" % (event1, event2),
                    file=sys.stderr)
                print(pos1, pos2)
                exit(1)

            if shipit and pos1 < pos2:
                chrom_segments[chrom].append((pos1, pos2))

    return chrom_segments


def break_large_contigs(contigs, break_t, verbose=False):
    """Break large contigs in half until all contigs are under
        the size threshold."""

    # initialize a heapq of contigs and lengths
    contig_heapq = []
    for ctg in contigs:
        ctg_len = ctg.end - ctg.start
        heapq.heappush(contig_heapq, (-ctg_len, ctg))

    ctg_len = break_t + 1
    while ctg_len > break_t:

        # pop largest contig
        ctg_nlen, ctg = heapq.heappop(contig_heapq)
        ctg_len = -ctg_nlen

        # if too large
        if ctg_len > break_t:
            if verbose:
                print('Breaking %s:%d-%d (%d nt)' % (ctg.chr,ctg.start,ctg.end,ctg_len))

            # break in two
            ctg_mid = ctg.start + ctg_len//2

            try:
                ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)
            except AttributeError:
                ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

            # add left
            ctg_left_len = ctg_left.end - ctg_left.start
            heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

            # add right
            ctg_right_len = ctg_right.end - ctg_right.start
            heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

    # return to list
    contigs = [len_ctg[1] for len_ctg in contig_heapq]

    return contigs


def write_seqs_bed(bed_file, seqs):
    """write sequences to BED file."""
    bed_out = open(bed_file, 'w')
    for i in range(len(seqs)):
        start = seqs[i].start
        end = seqs[i].end
        if start >= 0:
            line = '%s\t%d\t%d' % (seqs[i].chr, start, end)
            print(line, file=bed_out)
    bed_out.close()



def rejoin_large_contigs(contigs):
    """ Rejoin large contigs that were broken up before alignment comparison."""

    # split list by chromosome
    chr_contigs = {}
    for ctg in contigs:
        chr_contigs.setdefault(ctg.chr,[]).append(ctg)

    contigs = []
    for chrm in chr_contigs:
        # sort within chromosome
        chr_contigs[chrm].sort(key=lambda x: x.start)

        ctg_ongoing = chr_contigs[chrm][0]
        for i in range(1, len(chr_contigs[chrm])):
            ctg_this = chr_contigs[chrm][i]
            if ctg_ongoing.end == ctg_this.start:
                ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
            else:
                # conclude ongoing
                contigs.append(ctg_ongoing)

                # move to next
                ctg_ongoing = ctg_this

        # conclude final
        contigs.append(ctg_ongoing)

    return contigs


def contig_sequences(contigs, seq_length, stride, snap=1):
    ''' Break up a list of Contig's into a list of ModelSeq's. '''
    mseqs = []
    for ctg in contigs:
        seq_start = int(np.ceil(ctg.start/snap)*snap)
        seq_end = seq_start + seq_length

        while seq_end <= ctg.end:
            # record sequence
            mseqs.append(Contig(ctg.chr, seq_start, seq_end))
            # update
            seq_start += stride
            seq_end += stride
        
    return mseqs


def make_contigs(chr_list):
    '''genome-wide coverage. '''
    chrom_contigs = {}
    for chr in chr_list:
        chrom_contigs[chr] = [(0, chr_list[chr])]
    return chrom_contigs


def get_peak_regions(peak_paths_input_data, peak_paths_output_data, chr_list):
    """ 
    Define samples according to the peak of the target sequencing data

    Args:
        peak_paths_input_data: path of peak of input sequence data
        peak_paths_output_data: path of peak of target sequence data
        chr_list: list of prescribed chromosome lengths

    """
    temporary_path = os.path.join(os.path.dirname(__file__), 'temporary')
    if not os.path.isdir(temporary_path):
        os.mkdir(temporary_path)

    # add peaks to gap file
    for per_input_peak_path in peak_paths_input_data:
        peaks_region = os.path.join(temporary_path, 'peaks_region.bed')
        # if there are remnants from the previous run, delete
        if os.path.exists(peaks_region):
            os.remove(peaks_region)

        add_peak_cmd = 'cat ' + per_input_peak_path + ' >> ' + peaks_region
        p = subprocess.Popen(add_peak_cmd, shell=True)
        p.wait()

    for peak_path in peak_paths_output_data:
        output_peak_list = os.listdir(peak_path)
        for output_peak_file in output_peak_list:
            per_output_peak_path = os.path.join(peak_path, output_peak_file)
            add_peak_cmd = 'cat ' + per_output_peak_path + ' >> ' + peaks_region
            p = subprocess.Popen(add_peak_cmd, shell=True)
            p.wait()

    # sort
    sort_path = os.path.join(temporary_path, 'sorted_gap.bed')
    sort_cmd = 'sort-bed ' + peaks_region + ' > ' + sort_path
    p = subprocess.Popen(sort_cmd, shell=True)
    p.wait()

    # merge
    merge_path = os.path.join(temporary_path, 'merged_gap.bed')
    merge_cmd = 'bedtools merge -i ' + sort_path + ' > ' + merge_path
    p = subprocess.Popen(merge_cmd, shell=True)
    p.wait()

    # remove file
    # os.remove(sort_path)
    # os.remove(peaks_region)

    # change filename
    os.rename(merge_path, peaks_region)

    peaks = []

    # Only the peak in the candidate region is reserved
    for line in open(peaks_region):
        item = line.split()
        if item[0] in chr_list:
            if int(item[2]) <= chr_list[item[0]]:
                peaks.append(line.split())
  
    return peaks


def get_samples_except_peak(candidate_regions, peak_paths_input_data, peak_paths_output_data, seq_length=114688, overlap=81920):
    """ 
    negative samples

    Args:
        candidate_regions: candidate regions  dict:{chr:[(start0, end0), (start1, end1)], ...}
        peak_paths_input_data: path of peak of input sequence data
        peak_paths_output_data: path of peak of target sequence data
        seq_length: length of the sample
        overlap: the overlap between the starting points of the two samples

    Return:
        candidate_samples: list[num_samples, 3]
    """
    temporary_path = os.path.join(os.path.dirname(__file__), 'temporary')
    if not os.path.isdir(temporary_path):
        os.mkdir(temporary_path)

    stride = seq_length - overlap

    seed=88
    random.seed(seed)
    np.random.seed(seed)

    print(candidate_regions)
    chr_list = {}
    for chr in candidate_regions:
        chr_list[chr] = candidate_regions[chr][-1][1]

    chrom_contigs = make_contigs(chr_list)
    
    if len(peak_paths_input_data) != 0 and peak_paths_output_data != 0:
        peak_regions = get_peak_regions(peak_paths_input_data, peak_paths_output_data, chr_list)

        # remove peak regions
        chrom_contigs = remove_peak_regions(chrom_contigs, peak_regions)

    # ditch the chromosomes for contigs
    contigs = []
    for chrom in chrom_contigs:
        contigs += [Contig(chrom, ctg_start, ctg_end)
                    for ctg_start, ctg_end in chrom_contigs[chrom]]

    # filter for large enough
    contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_length]
    # break up large contigs
    break_t = 786432
    contigs = break_large_contigs(contigs, break_t)
      
    # print contigs to BED file
    ctg_bed_file = '%s/contigs_break.bed' % temporary_path
    write_seqs_bed(ctg_bed_file, contigs)

    # rejoin broken contigs within set
    contigs = rejoin_large_contigs(contigs)

    # write labeled contigs to BED file
    ctg_bed_file = '%s/contigs.bed' % temporary_path
    ctg_bed_out = open(ctg_bed_file, 'w')
    for ctg in contigs:
        line = '%s\t%d\t%d' % (ctg.chr, ctg.start, ctg.end)
        print(line, file=ctg_bed_out)
    ctg_bed_out.close()
     
    # stride sequences across contig
    mseqs = contig_sequences(contigs, seq_length, stride)

    # write sequences to BED
    seqs_bed_file = '%s/negatives.bed' % temporary_path
    write_seqs_bed(seqs_bed_file, mseqs)

    candidate_path = os.path.join(temporary_path, 'candidate.bed')
    candidate_write  = write_bed(candidate_regions, candidate_path)
    candidate_samples = select_candidate_samples(candidate_regions, candidate_path, seqs_bed_file, seq_length)

    # all negative samples use the first RO-seq file
    candidate_samples = add_key_values(candidate_samples, 'seq_num', 0)

    candidate_samples = convert_dict_2_bed_dataframe(candidate_samples)

    # remove temporary files
    shutil.rmtree(temporary_path)

    return np.asarray(candidate_samples)
    
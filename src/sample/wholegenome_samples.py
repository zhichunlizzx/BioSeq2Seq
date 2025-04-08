#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import numpy as np
from sample.candidate_region import split_contigs
from sample.negative_samples import Contig, contig_sequences


def get_predicted_samples(genome_region, include_chr, seq_length, except_region_file=None, except_chr=None, start_posi=None,):
  """ 
  get samples of whole genome

  Args:
    genome_region: the extent of each chromosome in the genome
    include_chr: chromosomal data needed for prediction
    seq_length: the length of each sample
    except_region_file: bed file for chromosomal regions not of interest
    stride: the interval between the starting points of the two samples
    except_chr: chromosome data not needed for prediction
  Return:
    predicted_samples: [num_of_samples, 3]
  """
  stride=seq_length

  predicted_region = {}
  if start_posi is None:
    if include_chr is not None:
      for chr in include_chr:
        predicted_region[chr]=genome_region[chr]
    elif except_chr is not None:
      for chr in genome_region:
        if not(chr in except_chr):
          predicted_region[chr]=genome_region[chr]
  else:
    
    if include_chr is not None:
      
      for chr in include_chr:
        predicted_region[chr]=[(start_posi, genome_region[chr][0][1])]
    elif except_chr is not None:
      for chr in genome_region:
        if not(chr in except_chr):
          predicted_region[chr]=[(start_posi, genome_region[chr][0][1])]

  
  if except_region_file is None:
    chrom_contigs = predicted_region
  else:
    chrom_contigs = split_contigs(predicted_region, except_region_file)

  # ditch the chromosomes for contigs
  contigs = []
  for chrom in chrom_contigs:
    contigs += [Contig(chrom, ctg_start, ctg_end)
                for ctg_start, ctg_end in chrom_contigs[chrom]]
    
  contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_length]

  # stride sequences across contig
  mseqs = contig_sequences(contigs, seq_length, stride)
  predicted_samples = []
  for i in range(len(mseqs)):
    predicted_samples.append([mseqs[i].chr, mseqs[i].start, mseqs[i].end, 0])
  return np.asarray(predicted_samples)

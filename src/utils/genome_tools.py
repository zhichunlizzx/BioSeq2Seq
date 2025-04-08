#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import os
import pyBigWig
import h5py
import sys
import pandas as pd
import numpy as np


class CovFace:
  def __init__(self, cov_file):
    self.cov_file = cov_file
    self.bigwig = False
    self.bed = False

    cov_ext = os.path.splitext(self.cov_file)[1].lower()
    if cov_ext == '.gz':
      cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

    if cov_ext in ['.bed', '.narrowpeak']:
      self.bed = True
      self.preprocess_bed()

    elif cov_ext in ['.bw','.bigwig']:
      self.cov_open = pyBigWig.open(self.cov_file, 'r')
      self.bigwig = True

    elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
      self.cov_open = h5py.File(self.cov_file, 'r')

    else:
      print('Cannot identify coverage file extension "%s".' % cov_ext,
            file=sys.stderr)
      exit(1)

  def preprocess_bed(self):
    # read BED
    bed_df = pd.read_csv(self.cov_file, sep='\t',
      usecols=range(3), names=['chr','start','end'])

    # for each chromosome
    self.cov_open = {}
    for chrm in bed_df.chr.unique():
      bed_chr_df = bed_df[bed_df.chr==chrm]

      # find max pos
      pos_max = bed_chr_df.end.max()

      # initialize array
      self.cov_open[chrm] = np.zeros(pos_max, dtype='bool')

      # set peaks
      for peak in bed_chr_df.itertuples():
        self.cov_open[peak.chr][peak.start:peak.end] = 1


  def read(self, chrm, start, end):
    if self.bigwig:
      # print(chrm, start, end)
      cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')
      # print(cov)

    else:
      if chrm in self.cov_open:
        cov = self.cov_open[chrm][start:end]
        pad_zeros = end-start-len(cov)
        if pad_zeros > 0:
          cov_pad = np.zeros(pad_zeros, dtype='bool')
          cov = np.concatenate([cov, cov_pad])
      else:
        print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
          (self.cov_file, chrm, start, end), file=sys.stderr)
        cov = np.zeros(end-start, dtype='float16')

    return cov

  def close(self):
    if not self.bed:
      self.cov_open.close()
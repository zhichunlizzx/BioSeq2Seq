#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
from utils.evaluation_tools import correlation_base_peak
import argparse

def call_correlation_base_tss(peak_file_dir,
                           predicted_file,
                           experiment_file,
                           include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                           'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                           'chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
                           length=1280,
                           ):
   """
   correlation of experiment and predicted files in four types of genome regions: (1) within 1k bp of TSS,
   (2) within 10k bp of TSS, (1) within 30k bp of TSS, (1) 30k bp away from TSS.
   Args:
      histone_list: the list contains the types of histone modifications that need to be evaluated.
      cellline: cell type.
      peak_file_dir: peak file path (histone_peak.sh生成)
      experiment_file: ChIP-seq histone modification file
      predicted_file: predicted histone modification file
   """

   peak_dir = peak_file_dir
   his = os.path.basename(predicted_file).split('.')[0]
   print(his)

   # near 1k
   path_peak = os.path.join(peak_dir, '1k_%s.bed' % his)

   try:
      correlation, spe = correlation_base_peak(predicted_file, experiment_file, path_peak, include_chr=include_chr, length=length)
   except:
      correlation, spe = -1, -1
   pearson_1k=correlation
   spearman_1k=spe

   # 1k-10k
   path_peak = os.path.join(peak_dir, '1k_10k_%s.bed' % his)
   correlation, spe = correlation_base_peak(predicted_file, experiment_file, path_peak, include_chr=include_chr, length=length)
   pearson_10k=correlation
   spearman_10k=spe

   # 10k-30k
   path_peak = os.path.join(peak_dir, '10k_30k_%s.bed' % his)
   correlation, spe = correlation_base_peak(predicted_file, experiment_file, path_peak, include_chr=include_chr, length=length)
   pearson_30k=correlation
   spearman_30k=spe

   # 30k++
   path_peak = os.path.join(peak_dir, 'away_30k_%s.bed' % his)
   correlation, spe = correlation_base_peak(predicted_file, experiment_file, path_peak, include_chr=include_chr, length=length)
   pearson_30k_=correlation
   spearman_30k_=spe

   print('0-1k pearson: %s' % pearson_1k)
   print('0-1k spearman %s' % spearman_1k)

   print('1-10k pearson %s' % pearson_10k)
   print('1-10k spearman %s' % spearman_10k)

   print('10-30k pearson %s' % pearson_30k)
   print('10-30k spearman %s' % spearman_30k)

   print('30k++ pearson %s' % pearson_30k_)
   print('30k++ spearman %s' % spearman_30k_)

   return pearson_1k, spearman_1k, pearson_10k, spearman_10k, pearson_30k, spearman_30k, pearson_30k_, spearman_30k_

if __name__ == '__main__':
   # Useage: python /local/zzx/code/BioSeq2Seq/src/HistoneModification/evaluation/correlation/base_TSS/corr_base_tss.py --exper /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.exper.bigWig --pre /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.pred.bw --tss /local/zzx/code/BioSeq2Seq/genome_regions/histone_peak_based_TSS/GM12878/near
   parser = argparse.ArgumentParser(description="Calculate correlation near Transcriptional Start Sites")
   parser.add_argument("--exper", dest="bw_ground_truth", type=str, help="Path to the ground truth bigWig file")
   parser.add_argument("--pre", dest="bw_predicted", type=str, help="Path to the predicted bigWig file")
   parser.add_argument("--tss", dest="TSS_file_dir", type=str, help="Path to the TSS file")
   parser.add_argument("--chr", dest="chromosome", default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'], nargs='*', help="Chromosomes for evaluation")
   parser.add_argument("-w", dest="peak_width", default=1280, type=int, help="peak width")
   args = parser.parse_args()

   include_chr = args.chromosome
   experiment_file = args.bw_ground_truth
   predicted_file = args.bw_predicted
   peak_width = args.peak_width
   
   if args.TSS_file_dir is None:
      TSS_file_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../../genome_regions/histone_peak_based_TSS/%s/near' % cellline))
   else:
      TSS_file_dir = args.TSS_file_dir

   call_correlation_base_tss(TSS_file_dir,
                  predicted_file,
                  experiment_file,
                  include_chr,
                  length=peak_width,
                  )
    



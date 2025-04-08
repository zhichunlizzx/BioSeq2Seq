import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
from utils.evaluation_tools import correlation_base_chromosome
import argparse


def correlation_genome_wide(
    predicted_file,
    experiment_file,
    include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
    window_size = 128,
):
    correlation_pearsonr, correlation_spearmanr = correlation_base_chromosome(predicted_file,
                                                                              experiment_file,
                                                                              include_chr,
                                                                              window_size
                                                                              )
    return correlation_pearsonr, correlation_spearmanr


if __name__ == '__main__':
    # Useage: python /local/zzx/code/BioSeq2Seq/src/HistoneModification/evaluation/correlation/Whole_Genome/corr_genome_wide.py --exper /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.exper.bigWig --pre /local/zzx/code/BioSeq2Seq/test_samples/H3k4me1.pred.bw --resolution 1280
    parser = argparse.ArgumentParser(description="Calculate correlation in whole genome wide")
    parser.add_argument("--exper", dest="bw_ground_truth", type=str, help="Path of the ground truth bigWig file")
    parser.add_argument("--pre", dest="bw_predicted", type=str, help="Path of the predicted bigWig file")
    parser.add_argument("--resolution", dest="resolution", default=128, type=int, help="Window size")
    parser.add_argument("--chr", dest="chromosome", default=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22'], nargs='*', help="Chromosome for evaluation")
    args = parser.parse_args()

    predicted_file = args.bw_ground_truth
    experiment_file = args.bw_predicted
    window_size=args.resolution
    chr_list = args.chromosome

    correlation_pearsonr, correlation_spearmanr = correlation_genome_wide(predicted_file, experiment_file, window_size=window_size, include_chr=chr_list)
    print('Pearson Corrlation: ', correlation_pearsonr)
    print('Spearman Corrlation: ', correlation_spearmanr)
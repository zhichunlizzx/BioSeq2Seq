import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
from utils.evaluation_tools import correlation_base_chromosome

def correlation_different_celllines(rna_seq_list: list,
                                    include_chr: list,
                                    window_size:int = 128,
                                    ):
    
    for i in range(len(rna_seq_list)):
        pearson_list = []
        spearman_list = []
        for j in range(len(rna_seq_list)):
            if j > i:
                continue
            elif j == i:
                pearson_list.append(1)
                spearman_list.append(1)
            else:
                print(os.path.basename(rna_seq_list[i]))
                print(os.path.basename(rna_seq_list[j]))

                pearson, separson = correlation_base_chromosome(rna_seq_list[j],
                                                                rna_seq_list[i],
                                                                window_size=window_size,
                                                                chr_list=include_chr
                                                                )
                pearson_list.append(pearson)
                spearman_list.append(separson)

        print('pearson: ', pearson_list)
        print('spearman: ', spearman_list)


if __name__ == '__main__':
    # Useage:
    rna_seq_list = [
        '/local/zzx/code/BioSeq2Seq/test_samples/rna/K562.bw',
        '/local/zzx/code/BioSeq2Seq/test_samples/rna/GM12878.bw',
        '/local/zzx/code/BioSeq2Seq/test_samples/rna/IMR90.bw',
        '/local/zzx/code/BioSeq2Seq/test_samples/rna/MCF7.bw',
        '/local/zzx/code/BioSeq2Seq/test_samples/rna/HepG2.bw'
    ]

    include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
    correlation_different_celllines(rna_seq_list, include_chr)
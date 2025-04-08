import os
from re import I
from dhit2_run import BioSeq2Seq, SamplePreprocess
import numpy as np
import shutil

chr_length = {'chr1':249250621,
            'chr2':243199373,
            'chr3':198022430,
            'chr4':191154276,
            'chr5':180915260,
            'chr6':171115067,
            'chr7':159138663,
            'chrX':155270560,
            'chr8':146364022,
            'chr9':141213431,
            'chr10':135534747,
            'chr11':135006516,
            'chr12':133851895,
            'chr13':115169878,
            'chr14':107349540,
            'chr15':102531392,
            'chr16':90354753,
            'chr17':81195210,
            'chr18':78077248,
            'chr20':63025520,
            'chrY':59373566,
            'chr19':59128983,
            'chr22':51304566,
            'chr21':48129895,
            }

reference_genome_file = '/local/zzx/data/hg19/hg19.fa'
input_data_peak_path = ['/local/zzx/za/Ro-seq-3/G1.bed',
                        '/local/zzx/za/Ro-seq-3/G2.bed',
                        '/local/zzx/za/Ro-seq-3/G3.bed',
                        '/local/zzx/za/Ro-seq-3/G4.bed',
                        '/local/zzx/za/Ro-seq-3/G5.bed',
                        '/local/zzx/za/Ro-seq-3/G6.bed',
                        '/local/zzx/za/Ro-seq-3/G7.bed']

output_data_peak_path = ['/local4/zzx/TF_maxATAC/K562/bed3']

sequence_data_file = [
                        [
                        # ['/local/zzx/hg19_data/proseq/Gm_minus.bw', '/local/zzx/hg19_data/proseq/Gm_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G1_minus.bw', '/local/zzx/hg19_data/proseq/G1_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G2_minus.bw', '/local/zzx/hg19_data/proseq/G2_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G3_minus.bw', '/local/zzx/hg19_data/proseq/G3_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G4_minus.bw', '/local/zzx/hg19_data/proseq/G4_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G5_minus.bw', '/local/zzx/hg19_data/proseq/G5_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G6_minus.bw', '/local/zzx/hg19_data/proseq/G6_plus.bw'],
                        ['/local/zzx/hg19_data/proseq/G7_minus.bw', '/local/zzx/hg19_data/proseq/G7_plus.bw'],
                         ]
                        
                        ]

tf_list = [
            'TCF7', 'NRF1', 'JUNB', 'NR2F6', 'RUNX1', 'ZBTB11',
            'ZBED1', 'MBD2', 'CREM', 'ETV6', 'SMAD5', 'SP1', 
            'NR2F1', 'RFX1', 'IKZF1', 'TCF7L2', 'ZKSCAN1', 'ZBTB33',
            'FOXA1', 'SREBF1', 'ZZZ3', 'CEBPZ', 'ELF1', 'ESRRA',
            'NKRF', 'FOXK2', 'ZBTB40', 'REST', 'PKNOX1', 'HES1',
            'NFXL1', 'ZNF407', 'NEUROD1', 'E2F8', 'POU5F1', 'ZNF282',
            'E4F1', 'ARNT', 'ASH1L', 'ZSCAN29', 'NFATC3', 'SMAD1',
            'ATF3', 'NFIC', 'SOX6', 'ATF2', 'ATF7', 'TCF12',
            'NR2C1', 'LEF1', 'ZNF24', 'GATAD2B', 'MNT', 'ELF4',
            'SKIL', 'FOXM1', 'ZNF592', 'MYBL2', 'EGR1', 'BHLHE40',
            'BACH1', 'JUND', 'RFX5', 'MAFF', 'MYC', 'ZNF274',
            'JUN', 'MXI1', 'TBP', 'CTCF', 'USF2', 'ATF1',
            'MAZ', 'MAFK', 'MAX', 'ZBTB7A', 'ETS1', 'FOSL1',
            'SETDB1', 'SIX5', 'MEF2A', 'TEAD4', 'CREB1', 'STAT5A',
            'NR2F2', 'CUX1', 'ZNF384', 'ELK1', 'SPI1', 'CEBPB']

tf_list = [
    'ARNT','ASH1L','ATF1','ATF7','BHLHE40','CREB1',
    'CREM','CUX1','E2F8','E4F1','EGR1','ELF1',
    'ELF4','ELK1','ESRRA','ETS1','ETV6','FOSL1',
    'FOXA1','FOXK2','FOXM1','GATAD2B','HES1','IKZF1',
    'JUN','JUNB','JUND','LEF1','MAFK','MAX',
    'MAZ','MBD2','MEF2A','MNT','MXI1','MYBL2',
    'MYC','NEUROD1','NFATC3','NFIC','NFXL1','NKRF',
    'NR2C1','NR2F1','NR2F2','NR2F6','NRF1','PKNOX1',
    'POU5F1','RFX1','RFX5','RUNX1','SETDB1','SIX5',
    'SKIL','SMAD1','SMAD5','SOX6','SP1','SREBF1',
    'STAT5A','TBP','TCF7','TCF7L2','TCF12','TEAD4',
    'USF2','ZBED1','ZBTB7A','ZBTB11','ZBTB40','ZKSCAN1',
    'ZNF24','ZNF274','ZNF282','ZNF407','ZNF592','ZSCAN29',
    'ZZZ3',     'ATF2','ATF3','BACH1','CEBPB','CEBPZ',
    'CTCF','MAFF','REST','SPI1','ZBTB33','ZNF384',
    ]
tf_list = ['ARNT']

print(tf_list)
init_lr=0.00001

tf_file = '/local4/zzx/TF_maxATAC/K562/macs2_narrow_peak'
target_list = []
target_seq_file = []
for tf in tf_list:
    target_seq_file.append(os.path.join(tf_file, '%s/%s.bw' % (tf, tf)))
    target_list.append(tf)

include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
              'chr18', 'chr19', 'chr20', 'chr21', 'chrX']
blacklist_file = '/local1/zzx/za/hg19Blacklist_1.bed'

sample_preprocess = SamplePreprocess(reference_genome_file=reference_genome_file,
                                sequencing_data_file=sequence_data_file,
                                include_chr=include_chr)



candidate_regions = sample_preprocess.get_candidate_regions()

samples = sample_preprocess.load_samples('/local/zzx/code/test_TF_2/TF_stage_samples_114k.bed')

train_samples = samples

blacklist = '/local/zzx/za/hg19Blacklist_1000k.bed'
test_samples = sample_preprocess.get_allgenome_samples(include_chr=['chr22'], blacklist_file=blacklist)
validation_samples = test_samples

dhit = BioSeq2Seq(reference_genome_file,
                  sequencing_data_file=sequence_data_file,
                  target_sequencing_file=target_seq_file)

model = dhit.build_model(target_list, nan=0, init_lr=init_lr)
# dhit.load_weights('/local1/zzx/code/test_TF_freezzing_2/model_save/1.4_with_DNase_all_cellline/best_model')

save_path = '/local/zzx/code/BioSeq2Seq/test_model'

dhit.train(train_samples,
        validation_samples,
        # init_lr=0.000001,
        epoch_num=200,
        step_per_epoch=10000,
        evaluation_epoch_num=1,
        valid_max_steps=100000,
        save_path=save_path,    
        lr_attenuation=1.5,
        lr_trans_epoch=10,
        promoter_gap_epoch=80,
        promoter_large_lr_epoch=120,
        )

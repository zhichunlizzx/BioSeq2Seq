import argparse
import pyBigWig
import os
from functions import load_chromosomes
from BioSeq2Seq_run import SamplePreprocess
from functions import predicted_to_bigwig
from model.HistoneModification.RD.model_RD import BioSeq2Seq as HM_RD_model
from model.FunctionalElement.RD.model_regre_two_inputs import BioSeq2Seq as FE_RD_model
from model.GeneExpression.model_regre_two_inputs import BioSeq2Seq as GE_model
from model.TFBS.model_regre_two_inputs_train_TF_multi_trunk import BioSeq2Seq as TFBS_model
import time
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def get_chr_length(bw_file, dna_file):
    bw = pyBigWig.open(bw_file)
    bw_chroms = bw.chroms()

    chr_length = {}
    dna_chroms = load_chromosomes(dna_file)

    for chrom in dna_chroms:
        if (not chrom.startswith('chrUn')) and ('_' not in chrom) and (chrom!='chrM') and (chrom!='chrEBV') and (chrom!='chrMT'):
            if dna_chroms[chrom][0][1] != bw_chroms[chrom]:
                print('################################################################################')
                print('Please provide the correct reference genome file.')
                print('################################################################################')
                exit()
            
            chr_length[chrom] = [[0, bw_chroms[chrom]],]
    return chr_length


def get_predicted_samples(reference_genome_file, sequence_data_file, include_chr):
    sample_preprocess = SamplePreprocess(reference_genome_file=reference_genome_file,
                            sequencing_data_file=sequence_data_file,
                            include_chr=include_chr)

    predicted_samples = sample_preprocess.get_evaluation_samples(include_chr=include_chr, blacklist_file=None)

    return predicted_samples


def get_model(model_type):
    current_path = os.path.dirname(__file__)
    if model_type == 'HistoneModification':
        model = HM_RD_model(channels=768, pooling_type='max')
        target_list = ['H3k122ac', 'H3k4me1', 'H3k4me2', 'H3k4me3', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k9ac', 'H3k9me3', 'H4k20me1',]
        weight_path = os.path.join(current_path, 'model/HistoneModification/RD/model.ckpt')
        model.load_weights(weight_path)
    elif model_type == 'FunctionalElement':
        target_list = ['promoter', 'genebody', 'polya', 'insulator']
        model = FE_RD_model(channels=768, output_channels=4, pooling_type='max')
        weight_path = os.path.join(current_path, 'model/FunctionalElement/RD/model.ckpt')
        print(weight_path)
        model.load_weights(weight_path)
    elif model_type == 'GeneExpression':
        model = GE_model(channels=768, output_channels=1, pooling_type='max')
        target_list = ['rna']
        weight_path = os.path.join(current_path, 'model/GeneExpression/model.ckpt')
        model.load_weights(weight_path)
    elif model_type == 'TFBS':
        model = TFBS_model(channels=768, output_channels=96, pooling_type='max')
        target_list = [
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
        weight_path = os.path.join(current_path, 'model/TFBS/model.ckpt')
        model.load_weights(weight_path)
    else:
        pass
    

    return model, target_list



def main():
    parser = argparse.ArgumentParser(description="Predict four types of transcriptional regulatory features based RO-seq amd DNA sequence")
    parser.add_argument("--multi", dest="multi", action="store_true", default=False, help="use muiti-GPU?")
    parser.add_argument("--minus", dest="minus", type=str, help="RO-seq minus file")
    parser.add_argument("--plus", dest="plus", type=str, help="RO-seq plus file")
    parser.add_argument("--op", dest="output_pre", type=str, default='BioSeq2Seq', help="Prefix for output file")
    parser.add_argument("-o", dest="outdir", type=str, help="Output directory")
    parser.add_argument("--ref", dest="ref_genome", type=str, help="reference genome (*.fa)")
    parser.add_argument("--type", dest="type", type=str, help="type of downstream task")
    args = parser.parse_args()
    start_time = datetime.now()
    print("Start time:", start_time.strftime('%Y-%m-%d %H:%M:%S'))

    minus_file = args.minus
    plus_file = args.plus
    outdir = args.outdir
    ref_genome = args.ref_genome
    model_type = args.type
    output_pre = args.output_pre

    work_path = os.getcwd()

    chr_length = get_chr_length(minus_file, ref_genome)
    chr_list = [chr for chr in chr_length]

    sequence_data_file = [[[minus_file, plus_file]]]

    # chr_list = ['chr22']

    predicted_samples = get_predicted_samples(ref_genome, sequence_data_file, chr_list)

    model, target_list = get_model(model_type)
    
    pred = predicted_to_bigwig(
                            model,
                            predicted_samples,
                            ref_genome,
                            sequence_data_file,
                            target_list,
                            chr_length,
                            outdir,
                            data_type='dna+seq',
                            extend=40960,
                            nan=0,
                            seq_length=114688,
                            window_size=128,
                            model_type=model_type,
                            pre_out=output_pre,
                        )
    
    print("The prediction is complete")
    end_time = datetime.now()
    print("End time:", end_time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
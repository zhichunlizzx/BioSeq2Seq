from utils.genome_tools import CovFace
from scipy.stats import pearsonr, spearmanr
from utils.bed_tools import read_peak_to_dict
import numpy as np
from model_function.functions import bw_2_chromosome_size

def correlation_base_peak(
                    predicted_file,
                    experiment_file,
                    path_peak,
                    length=None,
                    window_size=128,
                    include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                    ):
    """
    compute pearson correlation and spearman correlation in the region contains all peaks
    Args:
        length: specifies the width of peak
        window_size: bin size used in evaluation
    """
    predicted_open = CovFace(predicted_file)
    experiment_open = CovFace(experiment_file)
    
    peaks = read_peak_to_dict(path_peak, include_chr)

    pre_all = np.asarray([])
    exper_all = np.asarray([])

    for chr in peaks:
        if chr not in include_chr:
           continue
        chr_peaks = peaks[chr]
        for peak in chr_peaks:
            start = int(peak['start']) // window_size * window_size
            end = int(peak['end']) // window_size * window_size

            if length is not None:
                mid = (start + end) // 2
                start = mid - int(length/2)
                end = mid + int(length/2)

            if end == start:
                end = start + 1 * window_size

            predicted = predicted_open.read(chr, start, end)
            exper = experiment_open.read(chr, start, end)
                
            predicted[np.where(np.isnan(predicted))] = 1e-3
            predicted[np.where(np.isinf(predicted))] = 1e-3

            exper[np.where(np.isnan(exper))] = 1e-3
            exper[np.where(np.isinf(exper))] = 1e-3

            predicted = np.mean(predicted.reshape(-1, window_size), axis=-1)
            exper = np.mean(exper.reshape(-1, window_size), axis=-1)

            if (predicted == predicted[0]).all() or (exper == exper[0]).all():
                continue

            pre_all = np.append(pre_all, predicted)
            exper_all = np.append(exper_all, exper)

    pre_all = np.asarray(pre_all, dtype='float32')
    exper_all = np.asarray(exper_all, dtype='float32')

    correlation = round(pearsonr(pre_all, exper_all)[0], 4)
    spe = round(spearmanr(pre_all, exper_all)[0], 4)

    return correlation, spe


def correlation_base_chromosome(predicted_file, experiment_file, chr_list, window_size):
    """
    Calculate the correlation between a.bw and b.bw in terms of "window_size"
    Divide the genome into different small fragments according to the window_size, 
    calculate the average gene expression level of each small fragment, 
    and then calculate the correlation of the whole genome.

    Args:
            predicted_file: CovFace object of predicted.bw
            predicted_file: CovFace object of experiment.bw
            chr_list: a dict like {chromosome: 'lenth of chromosome'}
            window_size: bin size used in evaluation

    Output:
            (float) pearsonr correlation
            (float) spearman correlation
    """

    genome_cov_open_a = CovFace(predicted_file)
    genome_cov_open_b = CovFace(experiment_file)
    a = []
    b = []

    whole_genome_size = bw_2_chromosome_size(bw_file=predicted_file)

    for chr in chr_list:
        chr_length = whole_genome_size[chr][0][1]

        end = (chr_length - 1000000) - (chr_length - 2000000) % window_size

        chr_data_a = genome_cov_open_a.read(chr, 1000000, end)
        chr_data_b = genome_cov_open_b.read(chr, 1000000, end)


        chr_data_a[np.where(np.isnan(chr_data_a))] = 1e-3
        chr_data_b[np.where(np.isnan(chr_data_b))] = 1e-3

        chr_data_a[np.where(np.isinf(chr_data_a))] = 1e-3
        chr_data_b[np.where(np.isinf(chr_data_b))] = 1e-3
        
        chr_data_a = np.clip(chr_data_a, -384.0, 384.0)
        chr_data_b = np.clip(chr_data_b, -384.0, 384.0)

        chr_data_a = chr_data_a.reshape(-1, window_size)
        chr_data_a = np.mean(chr_data_a, axis=-1)

        chr_data_b = chr_data_b.reshape(-1, window_size)
        chr_data_b = np.mean(chr_data_b, axis=-1)

        if (chr_data_a == chr_data_a[0]).all() or (chr_data_b == chr_data_b[0]).all():
            continue

        a = np.hstack((a, chr_data_a))
        b = np.hstack((b, chr_data_b))

    a = np.asarray(a, dtype='float32')
    b = np.asarray(b, dtype='float32')


    correlation_pearsonr = round(pearsonr(a, b)[0], 4)
    correlation_spearmanr = round(spearmanr(a, b)[0], 4)
    return correlation_pearsonr, correlation_spearmanr
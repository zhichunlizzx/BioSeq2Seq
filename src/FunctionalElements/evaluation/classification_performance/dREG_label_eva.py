import os
import subprocess
from FunctionalElements.evaluation.classification_performance.classification_eva import read_peaks, write_peaks, peak_IOU, classified_eva
import inspect

def merge_bed_file(file_a, file_b, outfile, outpath):
    cmd_cat = 'cat ' + file_a + ' ' + file_b + ' > ' + outfile
    p = subprocess.Popen(cmd_cat, shell=True)
    p.wait()

    tmp_path = os.path.join(outpath, 'tmp.bed')
    cmd_sort = 'sort-bed ' + outfile + ' > ' + tmp_path
    p = subprocess.Popen(cmd_sort, shell=True)
    p.wait()

    cmd_merge = 'bedtools merge -i ' + tmp_path + ' > ' + outfile
    p = subprocess.Popen(cmd_merge, shell=True)
    p.wait()

    cmd_rm = ['rm', '-f', tmp_path]
    subprocess.call(cmd_rm)

    return 1


def eva(
        predicted_path,
        positive_label_path,
        negative_label_path,
        extend=50,
        include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
        threshold_peak=0.6,
        outpath=os.path.dirname(os.path.abspath(__file__)),
    ):
    eva_out = os.path.join(outpath, 'eva_results.txt')
    if os.path.exists(eva_out):
        os.remove(eva_out)

    target = 'dREG_promoter'

    # select base iou
    overlap_file = os.path.join(outpath, 'overlap_positive_%s.bed' % target)
    cmd_intersect = 'bedtools window -a ' + positive_label_path + ' -b ' + predicted_path + ' -w %d | bedtools overlap -i stdin -cols 2,3,5,6  > ' % extend + overlap_file
    p = subprocess.Popen(cmd_intersect, shell=True)
    p.wait()
    
    # 与label有overlap的预测peak，用于计算iou
    overlap_peaks = read_peaks(overlap_file, include_chr=include_chr)

    ###################################
    #iou 计算
    ###################################
    pre_iou = 0
    # 带有iou值的文件，只包含iou大于某个阈值的overlap peaks中的条目
    iou_path = os.path.join(outpath, 'iou_positive_%s.bed' % target)
    peak = overlap_peaks[0]
    peak.append(str(peak_IOU(peak, extend=extend)))

    iou_peaks = [peak]

    for peak in overlap_peaks[1:]:
        iou = peak_IOU(peak, extend=extend)
        # 一个label对应多个预测peak时，把iou加起来，然后赋值给这些peak
        if peak[:3] == iou_peaks[-1][:3]:
            iou += float(pre_iou)
            i = -1
            while iou_peaks[i][:3] == peak[:3] and i >= 0:
                iou_peaks[-1][-1] = str(iou)
                i -= 1
        
        peak.append(str(iou))
        iou_peaks.append(peak)
        pre_iou = iou

    filter_peak = []
    # 将使用iou过滤后的peak写入iou_path文件，并记录在filter_peak中
    with open(iou_path, 'w') as w_obj:
        for peak in iou_peaks:
            if float(peak[-1]) >= threshold_peak:
                filter_peak.append(peak)
                # print(float(peak[-3]))
                w_peak = ''
                for item in peak:
                    w_peak += item + '\t'
                w_peak = w_peak[:-1] + '\n'
                w_obj.write(w_peak)

    # write true positive to bed
    true_positive_file = os.path.join(outpath, 'true_positive_%s.bed' % target)
    with open(true_positive_file, 'w') as w_obj:
        pre_peak = filter_peak[0]
        w_obj.write(pre_peak[3] + '\t' + pre_peak[4] + '\t' + pre_peak[5] + '\n')
        for peak in filter_peak[1:]:
            # remove duplicates
            if [peak[3], peak[4], peak[5]] == [pre_peak[3], pre_peak[4], pre_peak[5]]:
                continue
            if float(peak[-1]) >= threshold_peak:
                w_obj.write(peak[3] + '\t' + peak[4] + '\t' + peak[5] + '\n')
            pre_peak = peak

    #########################
    # false positive
    #########################
    # 计算没有match上的label和预测peak
    nonintersect_label_path = os.path.join(outpath, 'nonintersect_label_%s.bed' % target)
    nonintersect_predicted_peak = os.path.join(outpath, 'false_positive_%s.bed' % target)

    # false negative and false positive
    bash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tp_fp.sh')
    cmd_fn_fp = ['bash', bash_path, positive_label_path, iou_path, nonintersect_label_path, nonintersect_predicted_peak, predicted_path]
    subprocess.call(cmd_fn_fp)
    # 只保留include chr的false positive
    write_peaks(nonintersect_predicted_peak, include_chr=include_chr)

    #########################
    # 计算tp fp
    #########################
    num_predicted_peak = len(read_peaks(predicted_path, include_chr=include_chr))
    fp = len(read_peaks(nonintersect_predicted_peak, include_chr=include_chr))
    tp = num_predicted_peak - fp


    #########################
    # 计算label查全率
    #########################
    num_label_peak = len(read_peaks(positive_label_path, include_chr=include_chr))
    nonintersect_label = read_peaks(nonintersect_label_path, include_chr=include_chr)
    num_nonintersect_label = len(nonintersect_label)
    
    num_intersect_label = num_label_peak - num_nonintersect_label

    #########################
    # negative TN FN
    #########################
    # 使用预测结果的补集作为反例，每个染色体正例反例数量相差1
    negative_path = os.path.join(outpath, 'negative_%s.bed' % target)
    idx_file = os.path.abspath(os.path.join(outpath, '../genome.idx'))

    cmd_subtract = 'bedtools subtract -a ' + idx_file + ' -b ' + predicted_path + ' > ' + negative_path
    p = subprocess.Popen(cmd_subtract, shell=True)
    p.wait()

    # 反例与label做overlap
    overlap_file = os.path.join(outpath, 'overlap_negative_%s.bed' % target)
    cmd_intersect = 'bedtools window -a ' + positive_label_path + ' -b ' + negative_path + ' -w 50 | bedtools overlap -i stdin -cols 2,3,5,6  > ' + overlap_file
    p = subprocess.Popen(cmd_intersect, shell=True)
    p.wait()

    overlap_peaks = read_peaks(overlap_file, include_chr=include_chr)

    pre_iou = 0
    iou_path = os.path.join(outpath, 'iou_negative_%s.bed' % target)

    peak = overlap_peaks[0]
    peak.append(str(peak_IOU(peak, extend=extend)))

    iou_peaks = [peak]

    for peak in overlap_peaks[1:]:
        iou = peak_IOU(peak, extend=extend)
        if peak[:3] == iou_peaks[-1][:3]:

            iou += float(pre_iou)
            i = -1
            while iou_peaks[i][:3] == peak[:3] and i >= 0:
                iou_peaks[-1][-1] = str(iou)
                i -= 1
        
        peak.append(str(iou))
        iou_peaks.append(peak)
        pre_iou = iou

    filted_peak = []
    # iou大于阈值的被视为假反例FN
    with open(iou_path, 'w') as w_obj:
        for peak in iou_peaks:
            if float(peak[-1]) >= threshold_peak:
                filted_peak.append(peak)
                # print(float(peak[-3]))
                w_peak = ''
                for item in peak:
                    w_peak += item + '\t'
                w_peak = w_peak[:-1] + '\n'
                w_obj.write(w_peak)

    # FN
    # remove duplicates
    rm_duc_filter_peak = [filted_peak[0]]

    # 去重
    for peak in filted_peak[1:]:
        if peak[3:6] != rm_duc_filter_peak[-1][3:6]:
            rm_duc_filter_peak.append(peak)
    fn = len(rm_duc_filter_peak)

    fn_file = os.path.join(outpath, 'false_negative_%s.bed' % target)
    with open (fn_file, 'w') as w_obj:
        for peak in rm_duc_filter_peak:
            w_obj.write("\t".join(peak[3:6]) + "\n")

    negatives = read_peaks(negative_label_path, include_chr=include_chr)

    tn = len(negatives) - fn


    print('TP:', tp)
    print('FP:', fp)
    print('TN:', tn)
    print('FN:', fn)
    print('intersect_label:', num_intersect_label)
    print('non_intersect_label:', num_nonintersect_label)

    acc, pre, recall, f_score = classified_eva(tn, fn, fp, tp)
    print('Acc:', acc)
    print('Pre:', pre)
    print('Recall:', recall)
    print('F-Score:', f_score)
    print('label recall:', num_intersect_label / (num_intersect_label + num_nonintersect_label))
    print('-'*50)


def eva_base_dREG_label(predicted_path,
                        positive_label_path,
                        negative_label_path,
                        outpath,
                        extend=50,
                        include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                        ):
    merge_label_path = os.path.join(outpath, 'merge_dREG_label.bed')
    merge_bed_file(positive_label_path, negative_label_path, merge_label_path, outpath=outpath)

    overlap_file = os.path.join(outpath, 'overlap_with_dREG.bed')
    tmp_bed = os.path.join(outpath, 'tmp.bed')
    cmd_intersect = 'bedtools window -a ' + merge_label_path + ' -b ' + predicted_path + ' -w %d | bedtools overlap -i stdin -cols 2,3,5,6  > ' % extend + overlap_file
    p = subprocess.Popen(cmd_intersect, shell=True)
    p.wait()

    cmd_cut = 'cut -f 4-6 ' + overlap_file + ' > ' + tmp_bed
    p = subprocess.Popen(cmd_cut, shell=True)
    p.wait()

    cmd_sort = 'sort-bed ' + tmp_bed + ' > ' + overlap_file
    p = subprocess.Popen(cmd_sort, shell=True)
    p.wait()

    cmd_merge = 'bedtools merge -i ' + overlap_file + ' > ' + tmp_bed + ' && ' + 'mv ' + tmp_bed + ' ' + overlap_file
    p = subprocess.Popen(cmd_merge, shell=True)
    p.wait()

    eva(overlap_file, positive_label_path, negative_label_path, include_chr=include_chr, outpath=outpath)


    
if __name__ == '__main__':
    current_file_path = os.path.dirname(inspect.getfile(inspect.currentframe()))

    path_fe_dir = os.path.join(current_file_path, '../../../../genome_regions/FE_file')
    path_test_dir = os.path.join(current_file_path, '../../../../test_samples/fe/out')

    outpath = os.path.abspath('%s/tre_evaluation/dREG' % path_test_dir)
    # predicted_path = os.path.abspath('%s/K562/promoter.bed' % path_fe_dir)
    predicted_path = os.path.abspath('%s/tre_evaluation/positive_promoter.bed' % path_test_dir)
    positive_label_path  = os.path.abspath('%s/dREG_label/K562_dREG_positive.bed' % path_fe_dir)
    negative_label_path = os.path.abspath('%s/dREG_label/K562_dREG_negative.bed' % path_fe_dir)

    # Useage:
    eva_base_dREG_label(predicted_path,
                        positive_label_path,
                        negative_label_path,
                        outpath,
                        include_chr=['chr22']
                        )



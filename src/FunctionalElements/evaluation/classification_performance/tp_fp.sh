# !/usr/bin/bash

label_file=$1
iou_file=$2
nonintersect_peak=$3
nonintersect_predicted_peak=$4
predicted_file=$5

# nonintersect peak
awk -F "\t" '!a[$1, $2, $3]++' $iou_file | bedtools subtract -a $label_file -b stdin > $nonintersect_peak

# false positive
awk -F "\t" '!a[$4, $5, $6]++' $iou_file | awk '{print $4 "\t" $5 "\t" $6}' | bedtools subtract -a $predicted_file -b stdin > $nonintersect_predicted_peak
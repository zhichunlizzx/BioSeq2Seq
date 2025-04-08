# !/usr/bin/bash
fdr_file=$1
filtered_fdr_file=$2
filtered_peak_file=$3
threshold=$4
# threshold=0.001

# echo 'threshold: '$threshold

awk -F "\t" '$4<'"$threshold"' {print}' $fdr_file > $filtered_fdr_file

awk -F "\t" '$4<'"$threshold"' {print $1 "\t" $2 "\t" $3}' $fdr_file > $filtered_peak_file
# !/usr/bin/bash

all_promoter_enhancer=$1
genebody=$2

tss_file="tss.bed"

awk -F "\t" '{if($5=="+") print $1 "\t" $2-1000 "\t" $2+1000; else if ($5=="-") print $1 "\t" $3-1000 "\t" $3+1000}' $genebody > $tss_file

# promoter
promoter_file="promoter.bed"
bedtools intersect -wa -a $all_promoter_enhancer -b $tss_file > $promoter_file
echo -e "\e[34mpromoter\e[0m file is saved in "$promoter_file

#enhancer
enhancer_file="enhancer.bed" 
bedtools intersect -v -a $all_promoter_enhancer -b $tss_file > $enhancer_file
echo -e "\e[34menhancer\e[0m  file is saved in "$enhancer_file

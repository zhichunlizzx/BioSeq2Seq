# !/usr/bin/bash

# useage
# ./histone_peak.sh /local/zzx/data/peak_file/histone_peak/k562_bed/merged/ /local2/zzx/hg19_data/dhit2_data/peak_file/TRE/test_genebody/ncbi/expression_stat/genebody_7_lines_only_gene/stat_percent_max_128_only_great_0.4/G1/stat_percent_max_128_only_great_0.4.bed /local2/zzx/hg19_data/dhit2_data/peak_file/histone_peak/new_histone_peak/K562/near wa

histone_path=$1
genebody_path=$2
outpath=$3

# itersect pattern
if [ -n "$4" ]
then
        intersect=$4
else
        intersect='v'
fi

echo $intersect

echo "select peak near tss 1k bp .."
# tss 1k
tss_outfile=$outpath"/tss_1k.bed"
temp_file=$outpath"/temp.bed"
# chr, start, end, gene type, +/-
awk -F "\t" '{if($5=="+") print $1 "\t" $2-500 "\t" $2+500; else if ($5=="-") print $1 "\t" $3-500 "\t" $3+500}' $genebody_path > $tss_outfile

# histone peak near tss
histone_file_list=$(ls $histone_path)
for histone in $histone_file_list
do
	histone_file=$histone_path"/"$histone
	out_histone_peak=$outpath"/1k_"$histone
	# echo $histone_file
	
	# histone peak near tss
	if [ -n "$4" ]
	then
		# echo "near tss"
		bedtools intersect -wa -a $histone_file -b $tss_outfile > $out_histone_peak
	else
		# histone peak away from tss
		# echo "away from tss"
		bedtools intersect -v -a $histone_file -b $tss_outfile > $out_histone_peak
	fi
	
	bedtools merge -i $out_histone_peak > $temp_file
	mv $temp_file $out_histone_peak
	
done

# cp $tss_outfile .
# rm $tss_outfile


echo "select peak near tss 1k-10k bp .."
# tss 1k-10k
tss_outfile=$outpath"/tss_10k.bed"
# chr, start, end, gene type, +/-
awk -F "\t" '{if($5=="+") print $1 "\t" $2-5000 "\t" $2+5000; else if ($5=="-") print $1 "\t" $3-5000 "\t" $3+5000}' $genebody_path > $tss_outfile

# histone peak near tss
histone_file_list=$(ls $histone_path)
for histone in $histone_file_list
do
	histone_file=$histone_path"/"$histone
	out_histone_peak_1k=$outpath"/1k_"$histone
	out_histone_peak_10k=$outpath"/10k_"$histone
	out_histone_peak_1k_10k=$outpath"/1k_10k_"$histone
	# echo $histone_file
	
	# histone peak near tss
	if [ -n "$4" ]
	then
		# echo "near tss"
		bedtools intersect -wa -a $histone_file -b $tss_outfile > $out_histone_peak_10k
	else
		# histone peak away from tss
		# echo "away from tss"
		bedtools intersect -v -a $histone_file -b $tss_outfile > $out_histone_peak_10k
	fi

	bedtools merge -i $out_histone_peak_10k > $temp_file
	mv $temp_file $out_histone_peak_10k

	bedtools subtract -a $out_histone_peak_10k -b $out_histone_peak_1k > $out_histone_peak_1k_10k

	bedtools merge -i $out_histone_peak_1k_10k > $temp_file
	mv $temp_file $out_histone_peak_1k_10k

done

# rm $tss_outfile


echo "select peak near tss 10k-30k bp .."
# tss 10k-30k
tss_outfile=$outpath"/tss_30k.bed"
# chr, start, end, gene type, +/-
awk -F "\t" '{if($5=="+") print $1 "\t" $2-15000 "\t" $2+15000; else if ($5=="-") print $1 "\t" $3-15000 "\t" $3+15000}' $genebody_path > $tss_outfile

# histone peak near tss
histone_file_list=$(ls $histone_path)
for histone in $histone_file_list
do
	histone_file=$histone_path"/"$histone
	out_histone_peak_10k=$outpath"/10k_"$histone
	out_histone_peak_30k=$outpath"/30k_"$histone
	out_histone_peak_10k_30k=$outpath"/10k_30k_"$histone
	out_histone_peak_away_30k=$outpath"/away_30k_"$histone
	# echo $histone_file
	
	# histone peak near tss
	if [ -n "$4" ]
	then
		# echo "near tss"
		bedtools intersect -wa -a $histone_file -b $tss_outfile > $out_histone_peak_30k
	else
		# histone peak away from tss
		# echo "away from tss"
		bedtools intersect -v -a $histone_file -b $tss_outfile > $out_histone_peak_30k
	fi
	
	bedtools merge -i $out_histone_peak_30k > $temp_file
	mv $temp_file $out_histone_peak_30k

	bedtools subtract -a $out_histone_peak_30k -b $out_histone_peak_10k > $out_histone_peak_10k_30k
	bedtools subtract -a $histone_file -b $out_histone_peak_30k > $out_histone_peak_away_30k

	bedtools merge -i $out_histone_peak_10k_30k > $temp_file
	mv $temp_file $out_histone_peak_10k_30k
	bedtools merge -i $out_histone_peak_away_30k > $temp_file
	mv $temp_file $out_histone_peak_away_30k
done

# rm $tss_outfile

# rm file
histone_file_list=$(ls $histone_path)
for histone in $histone_file_list
do
	out_histone_peak_10k=$outpath"/10k_"$histone
	out_histone_peak_30k=$outpath"/30k_"$histone
	rm -f $out_histone_peak_10k $out_histone_peak_30k
done
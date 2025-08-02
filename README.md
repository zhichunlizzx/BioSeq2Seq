[![language](https://img.shields.io/badge/language-Python-3776AB)](https://www.python.org/)
[![OS](https://img.shields.io/badge/OS-CentOS%20%7C%20Ubuntu-2C3E50)](https://www.centos.org/)
[![arch](https://img.shields.io/badge/arch-x86__64-blue)](https://en.wikipedia.org/wiki/X86-64)
[![GitHub last commit](https://img.shields.io/github/last-commit/zhichunlizzx/BioSeq2Seq)](https://github.com/zhichunlizzx/BioSeq2Seq/commits)

# BioSeq2Seq
This package provides an implementation for training, testing, and evaluation of the BioSeq2Seq framework.
![Hi](https://github.com/zhichunlizzx/BioSeq2Seq/blob/master/BioSeq2Seq.gateway.png?v=4&s=200 "dREG gateway")

## 🚀 About
**BioSeq2Seq** is a smart framework that allows users to provide, but not limited to, RO-seq and DNA sequences to predict a variety of transcriptional regulatory signals. Currently, BioSeq2Seq integrates four downstream analysis models for transcriptional regulation: histone modification prediction, functional element annotation, gene expression prediction, and transcriptional regulatory factor binding site (TFBS) prediction.

## 🔧 Install BioSeq2Seq

(1) Requirements
*   einops(0.4.1)
*   h5py(2.8.0)
*   pyBigWig(0.3.22)
*   pysam(0.19.0)
*   numpy(1.15.0)
*   tensorflow(2.4.0)

See `environment.yml`.

(2) Supported OS

Linux is supported at this time.

(3) Download BioSeq2Seq

Clone the BioSeq2Seq repository from GitHub

```shell
# download BioSeq2Seq
git clone https://github.com/zhichunlizzx/BioSeq2Seq.git
cd BioSeq2Seq
```

(4) Install BioSeq2Seq environment

Create the environment with the following command:

```shell
# create BioSeq2Seq environment
conda env create -f environment.yml -n BioSeq2Seq
# activate the environment
conda activate BioSeq2Seq
```

(5) Download the pretrained model weights

Download the pretrained model file from the following link:

👉 https://dreg.dnasequence.org/themes/dreg/assets/file/BioSeq2Seq_model.zip.

Then, move into the `BioSeq2Seq_Toolkit/BioSeq2Seq/model` directory and unzip the file:

```bash
cd BioSeq2Seq_Toolkit/BioSeq2Seq/model
# download the pretrained model
wget https://dreg.dnasequence.org/themes/dreg/assets/file/BioSeq2Seq_model.zip
# decompressing
unzip BioSeq2Seq_model.zip
cd ../..
```

## 📝 Data preparation

BioSeq2Seq requires double-stranded bigWig files and corresponding DNA sequences as input. 

(1) RO-seq bigWig file
The bigWig files must meet the following three criteria:

1. Reads must be mapped in point mode, using either the 5′ end (e.g., GRO-seq) or the 3′ end (e.g., PRO-seq). Do not represent reads as continuous regions starting from these ends. This mapping style differs from tools such as Tfit.

2. Each strand should contain only positive or only negative values, with no mixing of signs within a strand.

3. No normalization

As for how to generate bigWig files from fastq data, please refer to https://github.com/Danko-Lab/proseq2.0/.

(2) DNA sequence

Download the reference genomes (hg19 and mm10) from the following links:

*   hg19:
http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz

*   mm10:
http://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz

Then, navigate to your designated reference directory, download, and decompress the fasta files:

```bash
cd ./test_samples/ref/

# Download hg19 reference genome
wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
# Decompress hg19
gunzip hg19.fa.gz

cd ../..
```



## 🖥️ Prediction

BioSeq2Seq is an end-to-end transcriptional regulation analysis model that requires **RO-seq** data (including PRO-seq, GRO-seq, and ChRO-seq) from both the forward and reverse strands, along with the corresponding **reference genome**, to accurately predict various transcriptional regulatory activities. To use BioSeq2Seq, TensorFlow 2.4.0 or a later version must be properly installed. During operation, BioSeq2Seq will automatically utilize GPU acceleration to enhance computational performance.

Usage example:

```shell
bash --plus plus_strand.bw --minus minus_strand.bw  --ref ref.fa  -o outdir --type task_type --op out_prefix

Parameter explanations:
    --plus: plus strand bigWig file
    --minus: minus strand bigWig file
    --ref: reference genome FASTA
    --type: task type (HistoneModification, FunctionalElement, GeneExpression, TFBS)
    -o: output directory
    --op: output filename prefix (optional)
```

For example, to run BioSeq2Seq using RO-seq data of hg19, use:

```shell
bash --plus BioSeq2Seq/test_samples/roseq/plus.bw --minus BioSeq2Seq/test_samples/roseq/minus.bw  --ref BioSeq2Seq/test_samples/ref/hg19.fa  -o ./outdir --type HistoneModification --op histone
```

That command takes ~1-2 hours to execute on Ubuntu on a NVIDIA RTX 3090.

BioSeq2Seq outputs predicted signals in bigWig format. For functional element prediction tasks, you can use `fe_classfication_evaluation()` from `BioSeq2Seq/src/FunctionalElements/evaluation/classification_performance/classification_eva.py` to perform peak calling and evaluate the prediction performance.

Pre-trained model weights for different downstream tasks of BioSeq2Seq are available here: https://dreg.dnasequence.org/themes/dreg/assets/file/BioSeq2Seq_model.zip.


## 📝Train a new model

The training of a new model requires the following types of data:
*   Double-stranded RO-seq data ("xx_plus.bw, xx_minus.bw", optional — at least one of RO-seq or reference genome data must be provided)
*   Reference genome data ("hg19.fa", optional — at least one of reference genome or RO-seq data must be provided)
*   Target ground truth (such as histone modification ChIP-seq, RNA-seq, or other omics data)
*   Genome blacklist (optional)

When using a trained model for prediction, it is not necessary to provide the target ground truth. The detailed process of training and outputting prediction results can be found in `src/train.ipynb`.

## 📊 Evaluation
This package provides evaluation methods for four subtasks of BioSeq2Seq, see detail in `src/evaluation.ipynb`.

## 🧬 Targets of downstream tasks
|index|Downstream task|Item|Type|
|:-|:-|:-|:-|
|1|Histone modification|H3K4me1, H3K122ac, H3K4me2, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9ac, H3K9me3, H4K20me1|ChIP-seq|
|2|Functional element|Promoter, Insulator, Poly(A), Gene Body|annotation|
|3|Gene expression||RNA-seq|
|4|TFBS|TCF7, NRF1, JUNB, NR2F6, RUNX1, ZBTB11, ZBED1, MBD2, CREM, ETV6, SMAD5, SP1, NR2F1, RFX1, IKZF1, TCF7L2, ZKSCAN1, ZBTB33, FOXA1, SREBF1, ZZZ3, CEBPZ, ELF1, ESRRA, NKRF, FOXK2, ZBTB40, REST, PKNOX1, HES1, NFXL1, ZNF47, NEUROD1, E2F8, POU5F1, ZNF282, E4F1, ARNT, ASH1L, ZSCAN29, NFATC3, SMAD1, ATF3, NFIC, SOX6, ATF2, ATF7, TCF12, NR2C1, LEF1, ZNF24, GATAD2B, MNT, ELF4, SKIL, FOXM1, ZNF592, MYBL2, EGR1, BHLHE40, BACH1, JUND, RFX5, MAFF, MYC, ZNF274, CEBPB, MXI1, TBP, CTCF, USF2, ATF1, MAZ, MAFK, MAX, ZBTB7A, ETS1, FOSL1, SPI1, SIX5, MEF2A, TEAD4, CREB1, STAT5A, NR2F2, CUX1, ZNF384, ELK1, JUN, SETDB1, |TFBS peak|









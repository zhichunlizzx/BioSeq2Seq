[![language](https://img.shields.io/badge/language-Python-3776AB)](https://www.python.org/)
[![OS](https://img.shields.io/badge/OS-CentOS%20%7C%20Ubuntu-2C3E50)](https://www.centos.org/)
[![arch](https://img.shields.io/badge/arch-x86__64-blue)](https://en.wikipedia.org/wiki/X86-64)
[![GitHub release](https://img.shields.io/github/v/release/zhichunlizzx/BioSeq2Seq)](https://github.com/zhichunlizzx/BioSeq2Seq/releases)
[![GitHub release date](https://img.shields.io/github/release-date/zhichunlizzx/BioSeq2Seq)](https://github.com/zhichunlizzx/BioSeq2Seq/releases)
[![GitHub last commit](https://img.shields.io/github/last-commit/zhichunlizzx/BioSeq2Seq)](https://github.com/zhichunlizzx/BioSeq2Seq/commits)


# BioSeq2Seq
This package provides an implementation for training, testing, and evaluation of the BioSeq2Seq framework.

# Setup
Requirements:
*   einops(0.4.1)
*   h5py(2.8.0)
*   pyBigWig(0.3.22)
*   pysam(0.19.0)
*   numpy(1.15.0)
*   tensorflow(2.4.0)

See `environment.yml`.

Create the environment with the following command:

```shell
conda env create -f environment.yml -n my_env
```

# Training and outputs results
The training of the model requires the following types of data:
*   RO-seq double-stranded data (optional — at least one of RO-seq or reference genome data must be provided)
*   Reference genome data (optional — at least one of reference genome or RO-seq data must be provided)
*   Target ground truth (such as histone modification ChIP-seq, RNA-seq, or other omics data)
*   Genome blacklist (optional)

When using a trained model for prediction, it is not necessary to provide the target ground truth. The detailed process of training and outputting prediction results can be found in `train.ipynb`.

# Evaluation
This package provides evaluation methods for four subtasks of BioSeq2Seq, see detail in `evaluation.ipynb`.









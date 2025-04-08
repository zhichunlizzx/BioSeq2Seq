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









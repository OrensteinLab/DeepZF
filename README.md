# DeepZF
We present DeepZF, a two-step deep-learning-based pipeline for predicting binding ZFs and their DNA-binding preferences given only the amino acid sequence of a C2H2-ZF protein. To the best of our knowledge, we compiled the first in vivo dataset of binding and non-binding ZFs for training the first ZF-binding classifier; BindZFpredictor. We took advantage of both in vivo and in vitro datasets to learn the recognition code of ZF-DNA binding through transfer learning. Our newly developed model; PWMpredictor is the first to utilize deep learning for the task.

# BindZFpredictor
# Prerequisites
BindZFpredictor architecture is based on [ProteinBERT](https://github.com/nadavbra/protein_bert) which was implemented using Tensorflow.

By installing [ProteinBERT](https://github.com/nadavbra/protein_bert) you will get all BindZFpredictor requiremnts:

```bash
pip install protein-bert
```
# Usage 
After installing ProteinBERT you can update finetunning.py as in this git for saving predections.
```bash
cd path/to/BindZFpredictor/directory
data_name="${i}_zf_${i}_b"
(where i = 10k k= [0,10] see Data/BindZFpredictor folder)
f="path/to/BindZFpredictor/directory/${data_name}"
mkdir -p $f
mkdir -p ${f}/predictions
python3.6 main_bindzfpredictor.py -b_n ${data_name} -r 1 -p_add ${f} >> out
```

# PWMpredictor

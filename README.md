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
```bash
cd path/to/BindZFpredictor/directory
data_name="${i}_zf_${i}_b"
(where i = 10k k= $0\leq i\leq 10$)
f="ath/to/BindZFpredictor/directory/${data_name}"
```

# PWMpredictor

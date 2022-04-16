# DeepZF
We present DeepZF, a two-step deep-learning-based pipeline for predicting binding ZFs and their DNA-binding preferences given only the amino acid sequence of a C2H2-ZF protein. To the best of our knowledge, we compiled the first in vivo dataset of binding and non-binding ZFs for training the first ZF-binding classifier; BindZFpredictor. We took advantage of both in vivo and in vitro datasets to learn the recognition code of ZF-DNA binding through transfer learning. Our newly developed model; PWMpredictor is the first to utilize deep learning for the task.

# BindZFpredictor
## Prerequisites
BindZFpredictor architecture is based on [ProteinBERT](https://github.com/nadavbra/protein_bert) which was implemented using Tensorflow.

By installing [ProteinBERT](https://github.com/nadavbra/protein_bert) you will get all BindZFpredictor requiremnts:

```bash
pip install protein-bert
```
## Usage 
### Training BindZFpredictor
After installing ProteinBERT you can update finetunning.py as in this git for saving predections.

```bash
1. cd path/to/BindZFpredictor/directory

2.Creating saving folders 
   data_name="${i}_zf_${i}_b"
   (where i = 10k k= [0,10] see Data/BindZFpredictor folder)
   f="path/to/BindZFpredictor/directory/${data_name}"
   mkdir -p $f
   mkdir -p ${f}/predictions

3. Run model
   python3.6 main_bindzfpredictor.py -b_n ${data_name} -b_d path/to/bemchmark_directory -m_d path/to/ProteinBERT_pretrained_model -r 1 -p_add ${f} >> out
```
### Flags

```
   '-b_n', '--benchmark_name', help='zfs data and labels name ', type=str, required=True
   '-b_d', '--benchmark_dir', help='zfs data and labels directory ', type=str, required=True
    '-m_d', '--model_dir', help='ProteinBERT pretrained model directory', type=str, required=True
    '-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True
    '-p_add', '--pred_add', help='predictions saving folders add ', type=str, required=True
```
### Creating predicted binding ZF dataframe ans evaluate model
python3.6 create_zf_pred_df_and_cal_auc.py -p_add path/to/predected ZF -m_p path/to/Data

### Flags
```
'-p_add', '--pred_add', help='predictions saving folders add ', type=str, required=True
'-m_p', '--main_path', help='main path add ', type=str, required=True
```
# PWMpredictor

## Prerequisites


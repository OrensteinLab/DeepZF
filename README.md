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
* python >= 3.6
* tensorflow >= 2.4.0

## Usage
### Training PWMpredictor
1. create saving folders:

	f="path/to/PWMpredictor_directory"

	mkdir -p $f
	
	mkdir -p ${f}/history
	
	mkdir -p ${f}/models
	
	mkdir -p ${f}/predictions
  
2. run model:

python3.6 main_loo_PWMprecictor.py -d_add /path_to_data/ -add ${f} -zf_p_df zf_pred.csv -lr $lr -e $i -res_num 12 -r 0 -t_v retrain -ac_x False >> ${f}_out

### Flags
```
'-d_add', '--data_folder_address', help='main data and lables folder', type=str, required=True)
'-add', '--folder_address', help='main folder address for savings', type=str, required=True
'-zf_p_df', '--pred_zf_df', help='predicted binding zinc fingers df', type=str, required=True
'-lr', '--learning_rate', help='learning rate of adam optimizer', type=float, required=True
'-e', '--epochs', help='number of epochs', type=int, required=True
'-res_num', '--residual_num', help='number of residuals to use', type=int, required=True
'-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True
'-t_v', '--transfer_version', help='last_layer or retrain', type=str, required=True
'-ac_x', '--amino_acid_x', help='use b1h data with amino acid x', type=str, required=True
```

   

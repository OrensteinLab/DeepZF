# DeepZF
We present DeepZF, a two-step deep-learning-based pipeline for predicting binding ZFs and their DNA-binding preferences given only the amino acid sequence of a C2H2-ZF protein. To the best of our knowledge, we compiled the first in vivo dataset of binding and non-binding ZFs for training the first ZF-binding classifier; BindZFpredictor. We took advantage of both in vivo and in vitro datasets to learn the recognition code of ZF-DNA binding through transfer learning. Our newly developed model, PWMpredictor, is the first to utilize deep learning for the task.

# BindZFpredictor
## Prerequisites
BindZFpredictor architecture is based on [ProteinBERT](https://github.com/nadavbra/protein_bert) which was implemented using Tensorflow.

By installing [ProteinBERT](https://github.com/nadavbra/protein_bert) you will get all BindZFpredictor requirements:

```bash
pip install protein-bert
```

BindZFpredictor was run on python 3.6.8 with keras 2.6.0 and tensorflow 2.6.2

## Usage 

### Predicting using BindZFpredictor
Create trained model file model.p by concatenating its splits:
```
cd BindZF_predictor/code/

cat x?? > model.p
```

Run under BindZFpredictor/code dir:
```
python3.6 main_bindzfpredictor_predict.py -in <input_file> -out <output_file> -m model.p -e encoder.p -r <gpu-0/1>
```
Example:
```
python3.6 main_bindzfpredictor_predict.py -in 40_zf_40_b.csv -out results.tsv -m model.p -e encoder.p -r 1
```

Note that you will need to identify the zinc fingers according to the regular expression by yourself. Then, append each finger by its adajcent 40aa residues on each side, and proive them as input to the command line above to obtain their DNA-binding probabilities.

### Training BindZFpredictor
After installing ProteinBERT you can update finetunning.py as in this git for saving predictions.


1.
 ```
 cd path/to/BindZFpredictor/directory
```
2. Create saving folders 
```
   data_name="${i}_zf_${i}_b"
   (where i = 10k k= [0,10] see Data/BindZFpredictor folder)
   f="path/to/BindZFpredictor/directory/${data_name}"
   mkdir -p $f
   mkdir -p ${f}/predictions
```
3. Run model
```
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
### Creating predicted binding ZF dataframe and evaluate model
```
python3.6 create_zf_pred_df_and_cal_auc.py -p_add path/to/predicted ZF -m_p path/to/Data
```
### Flags
```
'-p_add', '--pred_add', help='predictions saving folders add ', type=str, required=True
'-m_p', '--main_path', help='main path add ', type=str, required=True
```
# PWMpredictor

## Prerequisites
* python >= 3.6
* tensorflow >= 2.4.0
* natsort >= 8.4.0
* pandas >= 2.0.3

	
## Usage

### Predicting using PWMpredictor

Run under PWMpredictor/code dir:
```
python3.6 main_PWMpredictor.py -in <input_file> -out <output_file> -m <model_file>
```
Example:
```
python3.6 main_PWMpredictor.py -in ../../Data/PWMpredictor/c_rc_df.csv -out predictions.txt -m ../models/transfer_model.h5
```

### Training PWMpredictor
1. Create saving folders:
```
	f="path/to/PWMpredictor_directory"

	mkdir -p $f
	
	mkdir -p ${f}/history
	
	mkdir -p ${f}/models
	
	mkdir -p ${f}/predictions
  ```
2. Run model:
```
	python3.6 main_loo_PWMpredictor.py -d_add /path_to_data/ -add ${f} -zf_p_df c_rc_df.csv -lr $lr -e $i -res_num 12 -r 0 -t_v retrain -ac_x False >> ${f}_out
```
### Flags
```
'-d_add', '--data_folder_address', help='main data and labels folder', type=str, required=True)
'-add', '--folder_address', help='main folder address for savings', type=str, required=True
'-zf_p_df', '--pred_zf_df', help='predicted binding zinc fingers df', type=str, required=True
'-lr', '--learning_rate', help='learning rate of adam optimizer', type=float, required=True
'-e', '--epochs', help='number of epochs', type=int, required=True
'-res_num', '--residual_num', help='number of residuals to use: 4, 7, 12', type=int, required=True 
'-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True
'-t_v', '--transfer_version', help='last_layer or retrain', type=str, required=True
'-ac_x', '--amino_acid_x', help='use b1h data with amino acid x', type=str, required=True
```

### Model evaluation:
For PWMpredictor evaluation, we computed the Pearson correlation of each quartet in the PWM matrix representing one position in the binding site.
The fallowing script calcultes the Pearson correlation and saves the Pearson correlation scv file
```
python3.6 eval_PWMpredictor.py -p_add /predictions_folder/ --c_rc_add /path/to/c_rc_df.csv -s_add /path_for savings >> out
```
### Flags
```
'-p_add', '--pred_folder_add', help='c_rc predictions folder add', type=str, required=True
'-c_rc_add', '--c_rc_add', help='c_rc data frame folder add', type=str, required=True
'-s_add', '--s_add', help='saving folder add', type=str, required=True
```

# DeepZF 
## Prerequisites
1. Running the model
	* python >= 3.6
	* tensorflow >= 2.4.0
2. Model evaluation
	In addition to the above:
	* [MoSBAT](https://github.com/csglab/MoSBAT) for DeepZF evaluation.
		
## Model training
1. Same as stage 1 in PWMpredictor

2. Run model:
```
	python3.6 main_loo_PWMpredictor.py -d_add /path_to_data/ -add ${f} -zf_p_df zf_pred.csv -lr $lr -e $i -res_num 12 -r 0 -t_v retrain -ac_x False >> ${f}_out
```
### Flags
Same as in PWMpredictor
(-zf_p_df is different now)

### Model evaluation:
For DeepZF evaluation we calculated the similarity of two motif pairs (on the predicted and experimentally based PWM), using [MoSBAT](https://github.com/csglab/MoSBAT).
Run bash script: (update paths)
```
./eval_DeepZF.sh
```
The script above creates 2 folder:
1. mosbat_input: includes mosbat input txt  files: ground ruth and predicted PWMs.
2. mosbat_output: includes: 
   * results.energy.correl.txt
   * correlation data frame: Pearson correlation score for each protein.
   * out_eval_mosbat.txt: a txt file with mean and std score.

# Cite
https://doi.org/10.1093/bioinformatics/btac469

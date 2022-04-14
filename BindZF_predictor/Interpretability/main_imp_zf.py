import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import pandas as pd
from utils_imp_zf import *
import argparse

# -b_n 40_zf_40_b -r 0 -aten_add C:/Users/User/PycharmProjects/zf_binding_pref/PROTEIN_BERT/loo_pred_results

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b_n', '--benchmark_name', help='zfs data and labels name ', type=str, required=True)
    parser.add_argument('-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True)
    parser.add_argument('-aten_add', '--attention_add', help='attentions saving folders add ', type=str, required=True)
    parser.add_argument('-exp_n', '--exp_name', help='experiment name', type=str, required=True)

    args = parser.parse_args()
    arguments = vars(args)

    return arguments

def main(args):
    if args["run_gpu"] == 1:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        # force the server to run on cpu and not on Gpu
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    BENCHMARK_NAME = args['benchmark_name']
    # A local (non-global) bianry output
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)


    # Loading the dataset
    BENCHMARKS_DIR = '/data/sofiaa/proteinBERT/loo_zf_data/'
    import os
    train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.csv' % BENCHMARK_NAME)

    train_set = pd.read_csv(train_set_file_path).dropna()
    train_set, valid_set = train_test_split(train_set, stratify = train_set['label'], test_size = 0.1, random_state = 0)


    print(f'{len(train_set)} training set records, {len(valid_set)} validation set records')


    # Loading the pre-trained model and fine-tuning it on the loaded dataset

    pretrained_model_generator, input_encoder = load_pretrained_model('/data/sofiaa/proteinBERT/proteinbert_models')

    # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
            get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ]

    finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
            seq_len = 512, batch_size = 32, max_epochs_per_stage=1, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
            lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-05, callbacks = training_callbacks)

    print('finished training')
    attention_df = pd.read_csv(args["exp_name"]).dropna()
    print('loaded attenshed df of shape')
    print(attention_df.shape)
    main_calculate_attention(model_generator, attention_df, args["attention_add"])
    print('finished main_calculate_attention')





    # main_calculate_attention(model_generator, attention_df, 'all_zf', args["attention_add"])
    # main_calculate_attention(model_generator, binding_zf_df, args["attention_add"])
    # main_calculate_attention(model_generator, non_binding_zf_df, args["attention_add"])
    # main_calculate_attention(model_generator, first_zf_df, 'first_zfs', args["attention_add"])
    # main_calculate_attention(model_generator, last_zf_df, 'last_zfs', args["attention_add"])
    # main_calculate_attention(model_generator, middle_zf_df, 'middle_zfs', args["attention_add"])

    # attention_df['len'] = attention_df['seq'].str.len()
    # # attention_df = train_set[train_set['len'] == 92]
    #
    # binding_zf_df = attention_df[attention_df['label'] == 1]
    # non_binding_zf_df = attention_df[attention_df['label'] == 0]
    # first_zf_df = pd.DataFrame(columns=["label", "seq", "groups"])
    # last_zf_df = pd.DataFrame(columns=["label", "seq", "groups"])
    # middle_zf_df = pd.DataFrame(columns=["label", "seq", "groups"])
    #
    # for i in attention_df['groups'].unique():
    #     tmp_df = attention_df[attention_df['groups'] == i]
    #     tmp_df.reset_index(drop=True, inplace=True)
    #     first_zf_df = first_zf_df.append(tmp_df.loc[0], ignore_index=True)
    #     last_zf_df = last_zf_df.append(tmp_df.loc[tmp_df.shape[0]-1], ignore_index=True)
    #     middle_zf_df = middle_zf_df.append(tmp_df.loc[1: tmp_df.shape[0]-2], ignore_index=True)
    #
    # main_path = 'C:/Users/User/PycharmProjects/zf_binding_pref/PROTEIN_BERT/viz_df/'
    # attention_df.to_csv(main_path + 'all_zfs.csv')
    # binding_zf_df.to_csv(main_path + 'binding_zfs.csv')
    # non_binding_zf_df.to_csv(main_path + 'non_binding_zfs.csv')
    # first_zf_df.to_csv(main_path + 'first_zfs.csv')
    # last_zf_df.to_csv(main_path + 'last_zfs.csv')
    # middle_zf_df.to_csv(main_path + 'middle_zfs.csv')
if __name__ == "__main__":
    args = user_input()
    main(args)



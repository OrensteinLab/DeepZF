import os
import tensorflow as tf
import pandas as pd
from IPython.display import display
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import argparse

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b_n', '--benchmark_name', help='zfs data and labels name ', type=str, required=True)
    parser.add_argument('-b_d', '--benchmark_dir', help='zfs data and labels directory ', type=str, required=True)
    parser.add_argument('-m_d', '--model_dir', help='ProteinBERT pretrained model directory', type=str, required=True)
    parser.add_argument('-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True)
    parser.add_argument('-p_add', '--pred_add', help='predictions saving folders add ', type=str, required=True)


    args = parser.parse_args()
    arguments = vars(args)

    return arguments

def main(args):
    if args["run_gpu"] == 1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        # force the server to run on cpu and not on Gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    BENCHMARK_NAME = args['benchmark_name']
    # A local (non-global) bianry output
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

    # Loading the dataset
    BENCHMARKS_DIR = args['benchmark_dir']

    data_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.csv' % BENCHMARK_NAME)
    data_set = pd.read_csv(data_set_file_path).dropna()
    n_split = data_set['groups'].unique().shape[0]
    for i in range(n_split):
        print('Number of iteration %d' % (i))
        train_set = data_set[data_set['groups'] != i]

        train_set, valid_set = train_test_split(train_set, stratify=train_set['label'], test_size=0.1, random_state=0)

        test_set = data_set[data_set['groups'] == i]

        print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

        # Loading the pre-trained model and fine-tuning it on the loaded dataset
        pretrained_model_generator, input_encoder = load_pretrained_model(args['model_dir'])


        # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
        model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
                get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

        training_callbacks = [
            keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
            keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
        ]

        finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'],\
                seq_len = 512, batch_size = 32, max_epochs_per_stage=40, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
                lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-05, callbacks = training_callbacks)


        # Evaluating the performance on the test-set

        results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], \
                i, args["pred_add"], start_seq_len = 512, start_batch_size = 32, )

        print('Test-set performance:')
        display(results)

        print('Confusion matrix:')
        display(confusion_matrix)


if __name__ == "__main__":
    args = user_input()
    main(args)

import os
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from IPython.display import display
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.finetuning import encode_Y, encode_dataset, split_dataset_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import argparse
ADDED_TOKENS_PER_SEQ = 2


def predict_by_len(model_generator, input_encoder, output_spec, seqs, raw_Y, start_seq_len = 512, start_batch_size = 32, increase_factor = 2):
    
    assert model_generator.optimizer_weights is None
    
    dataset = pd.DataFrame({'seq': seqs, 'raw_y': raw_Y})
        
    results = []
    results_names = []
    y_trues = []
    y_preds = []
    
    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len(dataset, start_seq_len = start_seq_len, start_batch_size = start_batch_size, \
            increase_factor = increase_factor):

        X, y_true, sample_weights = encode_dataset(len_matching_dataset['seq'], len_matching_dataset['raw_y'], input_encoder, output_spec, \
                seq_len = seq_len, needs_filtering = False)
        
        assert set(np.unique(sample_weights)) <= {0.0, 1.0}
        y_mask = (sample_weights == 1)
        
        model = model_generator.create_model(seq_len)
        y_pred = model.predict(X, batch_size = batch_size)
        
        y_true = y_true[y_mask].flatten()
        y_pred = y_pred[y_mask]
        
        if output_spec.output_type.is_categorical:
            y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        else:
            y_pred = y_pred.flatten()
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        
    y_true = np.concatenate(y_trues, axis = 0)
    y_pred = np.concatenate(y_preds, axis = 0)
    
    return y_pred

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_file', help='zfs data', type=str, required=True)
    parser.add_argument('-out', '--output_file', help='results', type=str, required=True)
    parser.add_argument('-m', '--model_file', help='trained model', type=str, required=True)
    parser.add_argument('-e', '--encoder_file', help='encoder', type=str, required=True)
    parser.add_argument('-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True)

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


    INPUT = args['input_file']
    # A local (non-global) bianry output
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

    # Loading the dataset
    OUTPUT = args['output_file']

    data_set_file_path = os.path.join("./", args['input_file'])
    data_set = pd.read_csv(data_set_file_path).dropna()
    test_set = data_set

    # Loading trained model
    model_generator = pickle.load(open(args['model_file'], "rb"))
    input_encoder = pickle.load(open(args['encoder_file'], "rb"))

    # Predicting on the test-set
    results = predict_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], start_seq_len = 512, start_batch_size = 32, increase_factor = 2)

    # Saving predictions to a file
    np.savetxt(args['output_file'], results, delimiter='\t')


if __name__ == "__main__":
    args = user_input()
    main(args)

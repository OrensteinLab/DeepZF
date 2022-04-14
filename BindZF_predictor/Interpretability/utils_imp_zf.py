import numpy as np
from proteinbert import load_pretrained_model


def calculate_attentions(model, input_encoder, seq, seq_len=None):
    from tensorflow.keras import backend as K
    from proteinbert.tokenization import index_to_token

    if seq_len is None:
        seq_len = len(seq) + 2

    X = input_encoder.encode_X([seq], seq_len)
    (X_seq,), _ = X
    seq_tokens = list(map(index_to_token.get, X_seq))

    model_inputs = [layer.input for layer in model.layers if 'InputLayer' in str(type(layer))][::-1]
    model_attentions = [layer.calculate_attention(layer.input) for layer in model.layers if
                        'GlobalAttention' in str(type(layer))]
    invoke_model_attentions = K.function(model_inputs, model_attentions)
    attention_values = invoke_model_attentions(X)

    attention_labels = []
    merged_attention_values = []

    for attention_layer_index, attention_layer_values in enumerate(attention_values):
        for head_index, head_values in enumerate(attention_layer_values):
            attention_labels.append('Attention %d - head %d' % (attention_layer_index + 1, head_index + 1))
            merged_attention_values.append(head_values)

    attention_values = np.array(merged_attention_values)

    return attention_values, seq_tokens, attention_labels


def main_calculate_attention(model_generator, test_set, attention_add):
    IDEAL_LEN = 92
    pretrained_model_generator, input_encoder = load_pretrained_model('/data/sofiaa/proteinBERT/proteinbert_models')
    pretrained_model = pretrained_model_generator.create_model(IDEAL_LEN+2)
    model = model_generator.create_model(IDEAL_LEN+2)
    pretrained_attention_values_l = []
    finetuned_attention_values_l = []
    for i in range(1230, test_set.shape[0]):
        chosen_index = i
        print('chosen index is: %d' % (i))

        seq = test_set.loc[chosen_index, 'seq']
        label = test_set.loc[chosen_index, 'label']

        seq_len = len(seq) + 2
        if len(seq) < IDEAL_LEN:

            pretrained_model_tmp = pretrained_model_generator.create_model(seq_len)
            model_tmp = model_generator.create_model(seq_len)

            pretrained_attention_values, pretrained_seq_tokens, pretrained_attention_labels = calculate_attentions(
                pretrained_model_tmp,
                input_encoder,
                seq, \
                seq_len=seq_len)
            finetuned_attention_values, finetuned_seq_tokens, finetuned_attention_labels = calculate_attentions(model_tmp,
                                                                                                                input_encoder,
                                                                                                                seq, \
                                                                                                                seq_len=seq_len)

        else:
            pretrained_attention_values, pretrained_seq_tokens, pretrained_attention_labels = calculate_attentions(pretrained_model,
                                                                                                                   input_encoder,
                                                                                                                   seq, \
                                                                                                                   seq_len=seq_len)
            finetuned_attention_values, finetuned_seq_tokens, finetuned_attention_labels = calculate_attentions(model,
                                                                                                                input_encoder, seq, \

                                                                                                                seq_len=seq_len)
        finetuned_attention_values = 100 * (finetuned_attention_values).sum(axis=0)
        pretrained_attention_values = 100 * pretrained_attention_values.sum(axis=0)
        if len(seq) < IDEAL_LEN:
            pretrained_attention_values = np.concatenate((pretrained_attention_values, np.zeros(IDEAL_LEN - len(seq))))
            finetuned_attention_values = np.concatenate((finetuned_attention_values, np.zeros(IDEAL_LEN - len(seq))))

        np.save(attention_add + '/pre_train/' + 'pretrained_attention_values_' + str(chosen_index), pretrained_attention_values)
        np.save(attention_add + '/fine_tuned/' + 'finetuned_attention_values_' + str(chosen_index), finetuned_attention_values)

        # pretrained_attention_values_l.append(100 * pretrained_attention_values.sum(axis=0))
        # finetuned_attention_values_l.append(100 * (finetuned_attention_values - pretrained_attention_values).sum(axis=0))

        assert finetuned_seq_tokens == pretrained_seq_tokens
        assert finetuned_attention_labels == pretrained_attention_labels[:len(finetuned_attention_labels)]
    # np.save(attention_add + '/' + name + '_pretrained_attention_values', np.asarray(pretrained_attention_values_l))
    # np.save(attention_add + '/' + name + '_finetuned_attention_values', np.asarray(finetuned_attention_values_l))
    return

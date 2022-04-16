import random
from functions import *
import tensorflow as tf
import shap



def random_zf(res):

    return ''.join((random.choice('ACDEFGHIKLMNPQRSTVWY') for x in range(res)))

def create_input_model(df, res_num):
    if res_num == 12:
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_12'])

    return oneHot_C_RC_amino

res = 12
N = 100000
random_zf_l = []
for i in range(N):
    random_zf_l.append(random_zf(res))

random_zf_df = pd.DataFrame(random_zf_l, columns=['res_12'])
one_hot_c_rc_12res = oneHot_Amino_acid_vec(random_zf_df['res_12'])

model_path = '/data/sofiaa/pipeline_transfer_learning/b1h_7_pos/interpretability/res_12/all_data/retrain/all_data_lr_0.001_epocs_50/models/'
tl_model = tf.keras.models.load_model(model_path + 'transfer_model.h5')

data_path = '/data/sofiaa/data/'
c_rc_df = pd.read_csv(data_path + 'c_rc_df.csv', sep=' ')
data_input_model = create_input_model(c_rc_df, res)

x_train = data_input_model
x_test = one_hot_c_rc_12res


# compute SHAP values
explainer = shap.DeepExplainer(tl_model, x_train)
print('calculated explainer')
shap_values = explainer.shap_values(x_test)
print(shap_values.__len__())
path = '/data/sofiaa/pipeline_transfer_learning/b1h_7_pos/interpretability/res_12/shap_values'
for i in range(res):
    np.save(path + '/shap_values_' + str(i), np.mean(shap_values[i], axis=0))


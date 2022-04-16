
from functions import *
import functions_C_RC as f_c_rc
import RNN_functions


c_rc_df = pd.read_excel('C:/Users/User/Desktop/projects/Amino_DNA/C_RC/C_RC_data.xlsx')
c_rc_df = f_c_rc.group_zf_by_protein_name(c_rc_df)
zf_data_df, c_rc_df = RNN_functions.adjust_c_rc_zf_data('C:/Users/User/Desktop/projects/Amino_DNA/zf_connection_pred/', c_rc_df)
c_rc_df.rename(columns={"12_seq": "res_12"}, inplace=True)


c_rc_df['res_7'] = c_rc_df.res_12.map(lambda x: x[2] + x[4:9] + x[-1])
c_rc_df['res_7_b1h'] = c_rc_df.res_12.map(lambda x: x[5:])
c_rc_df['res_4'] = c_rc_df.res_12.map(lambda x: x[5] + x[7:9] + x[-1])

zf_data_df['res_12'] = zf_data_df['zf_seq']
zf_data_df['res_7'] = zf_data_df.res_12.map(lambda x: x[2] + x[4:9] + x[-1])
zf_data_df['res_7_b1h'] = zf_data_df.res_12.map(lambda x: x[5:])
zf_data_df['res_4'] = zf_data_df.res_12.map(lambda x: x[5] + x[7:9] + x[-1])


"one hot encoding"
one_hot_c_rc_4res = oneHot_Amino_acid_vec(c_rc_df['res_4'])
one_hot_c_rc_7res = oneHot_Amino_acid_vec(c_rc_df['res_7'])
one_hot_c_rc_7b1h_res = oneHot_Amino_acid_vec(c_rc_df['res_7_b1h'])
one_hot_c_rc_12res = oneHot_Amino_acid_vec(c_rc_df['res_12'])

"labels: the pwm is the same to all sequence residuals"
pwm = (c_rc_df.filter(items=['A1', 'C1', 'G1', 'T1', 'A2', 'C2', 'G2', 'T2', 'A3', 'C3', 'G3', 'T3'])).values
"Savings"
path = '/Transfer_learning/data_labels/'
np.save(path + 'ground_truth_c_rc', pwm)
np.save(path + 'onehot_encoding_c_rc_4res', one_hot_c_rc_4res)
np.save(path + 'onehot_encoding_c_rc_7res', one_hot_c_rc_7res)
np.save(path + 'onehot_encoding_c_rc_7b1hres', one_hot_c_rc_7b1h_res)
np.save(path + 'onehot_encoding_c_rc_12res', one_hot_c_rc_12res)
c_rc_df.to_csv(path + 'c_rc_df.csv', sep=' ', index=False)
zf_data_df.to_csv(path + 'zf_data_df.csv', sep=' ', index=False)
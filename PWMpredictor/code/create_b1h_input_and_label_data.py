from functions import *

"""This function creates B1H input and label for Transfer learning model:
INPUT DATA: each amino acid is represented by a 1X20 one hot vector therefore
             zinc finger with 12 positions is represented by 1X240 vector
             zinc finger with 7 positions is represented by 1X140 vector and
             we pad each finger to be a 12 positions long therefore final representation is 1x240
 LABEL: the model label is one a position weight matrix (pwm): 3X4 (3 possible positions and 4 DNA nucleotides),
        In our model the pwm is reshaped to  1X12 vector"""



B1H_one_finger_add = "UPDATE/one_finger_pwm_gt.txt"
file = open(B1H_one_finger_add)  # open data file

mat_l = []
prot_7_res_seq_l = []  # 7 residuals list
prot_4_res_seq_l = []  # 4 residuals list

lines = file.readlines()
for i in range(0, lines.__len__()-2, 8):
    mat_l.append(lines[i+3: i+4+4])
    prot_4_res_seq_l.append(lines[i+1][:-1])
    prot_7_res_seq_l.append(lines[i+2][:-1])

file.close()

"create pwm matrix (label)"
mat_pd = pd.DataFrame(mat_l).drop(columns=4)
mat_pd = mat_pd.applymap(lambda x: x[4: -1])
mat_pd['all'] = mat_pd[0] + ' ' + mat_pd[1] + ' ' + mat_pd[2] + ' ' + mat_pd[3]
mat_pd = mat_pd.applymap(lambda x: np.fromstring(x, dtype=float, sep=' '))
pwm = np.stack(mat_pd['all'])
reorder_index = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
pwm = pwm[:, reorder_index]

"create input data_frame: one hot encoding without amino acid X:"
"each amino acid is a binary 20 length vector"
string = 'XXXXX'
prot_7_res_df = pd.DataFrame(prot_7_res_seq_l, columns={'7_res'})
prot_12_res_df= prot_7_res_df.apply(lambda x: string + x['7_res'], axis=1)

prot_4_res_df = pd.DataFrame(prot_4_res_seq_l)


"save model input and label of data including amino acid X"
"amino acid X is encoded as a 20 length vector with probability 1/20"
one_hot_12res = oneHot_Amino_acid_vec(prot_12_res_df)
save_path = '/Transfer_learning/data_labels/'
np.save(save_path + 'ground_truth_b1h_pwm_12res', pwm)
np.save(save_path + 'onehot_encoding_b1h_12res', one_hot_12res)

""" one hot encoding for sequences without amino acid X"""

def find_X_amino_acid_index(prot_pd):
    """There are some protein sequences with X as amino acid, this function finds this indexes"""
    x_index_l = []
    for i in range(prot_pd.shape[0]):
        if "X" in prot_pd[0][i]:
            x_index_l.append(i)
    return x_index_l


x_index_4res_l = find_X_amino_acid_index(prot_4_res_df)
x_index_7res_l = find_X_amino_acid_index(prot_7_res_df)

# drop from data dataframe the protein sequences with amino acid X
prot_4_res_df.drop(x_index_4res_l, inplace=True)
prot_7_res_df.drop(x_index_7res_l, inplace=True)

"find one hot representation: each protein representd by on hot vector"
one_hot_4res = oneHot_Amino_acid_vec(prot_4_res_df[0])
one_hot_7res = oneHot_Amino_acid_vec(prot_7_res_df[0])

"update pwm matrix"
pwm_4res = np.delete(pwm, x_index_4res_l, axis=0)
pwm_7res = np.delete(pwm, x_index_7res_l, axis=0)


np.save(save_path + 'ground_truth_b1h_pwm_4res', pwm_4res)
np.save(save_path + 'ground_truth_b1h_pwm_7res', pwm_7res)
np.save(save_path + 'onehot_encoding_b1h_4res', one_hot_4res)
np.save(save_path + 'onehot_encoding_b1h_7res', one_hot_7res)

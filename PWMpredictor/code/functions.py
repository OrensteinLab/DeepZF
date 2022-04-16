import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
import os
# import matplotlib.pyplot as plt  # Matlab-style plotting

def ht_one_hot_encode_amino_acids(files_list):
    """this function is responsible to produce the OneHot
    encoding for the AMINO ACIDS strings.
    The OneHOt matrix is used for the convolutional process"""
    return list(map(one_hot_encoding_amino_acids, files_list))


def one_hot_encoding_amino_acids(sequence):
    # define universe of possible input values
    amino_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(amino_alphabet))

    # integer encode iht_one_hot_encode_amino_acidsnput data
    integer_encoded = [char_to_int[char] for char in sequence]

    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        if value == 20:
            onehot_encoded.append(list(0.05 * np.ones(20)))
        else:
            letter = [0 for _ in range(len(amino_alphabet)-1)]
            letter[value] = 1
            onehot_encoded.append(letter)
    return np.asarray(onehot_encoded)


def ht_one_hot_encode_DNA(files_list):
    """this function is responsible to produce the OneHot
    encoding for the DNA strings.
    The OneHOt matrix is used for the convolutional process"""
    return list(map(one_hot_encoder_DNA, files_list))


def one_hot_encoder_DNA(DNA_strings):
    """this function is responsible to produce the OneHot
    encoding for the DNA strings.
    The OneHOt matrix is used for the convolutional process"""

    DNA_strings = DNA_strings + "ACGT"
    trantab = DNA_strings.maketrans('ACGT', '0123')  # translate to matrix elements
    data = list(DNA_strings.translate(trantab))
    return to_categorical(data)[0:-4]


def dic_unique_amino_acids(amino_seq):
    val = np.arange(0, len(amino_seq) - 1)
    dic = dict((val, i) for i, val in enumerate(amino_seq))
    return dic


def oneHot_DNA_vec(dna_seq):
    """this function is responsible to produce the OneHot vector
    encoding for the DNA strings."""
    one_hot_dna_mat = ht_one_hot_encode_DNA(dna_seq)
    one_hot_dna_vec = []
    for array in one_hot_dna_mat:
        one_hot_dna_vec.append(array.flatten())
    return np.array(one_hot_dna_vec)


def oneHot_Amino_acid_vec(amino_seq):
    """this function is responsible to produce the OneHot vector
    encoding for the amino acids strings."""
    one_hot_matrix_amino_acids = ht_one_hot_encode_amino_acids(amino_seq)
    one_hot_amino_acids_vec = []

    # create a matrix: each row represents Amino acid
    for array in one_hot_matrix_amino_acids:
        one_hot_amino_acids_vec.append(array.flatten())
    return np.array(one_hot_amino_acids_vec)

def concat(one_hot_amino, one_hot_dna):
    """this function is responsible to concatenate the amino acids with dna. Each amino acid is concatenated with
    one of the triples of the dna."""

    # duplicate each dna as the number of amino acids
    dup_dna = np.repeat(one_hot_dna, repeats=len(one_hot_amino), axis=0)

    # duplicate the amino acid matrix as the number of dna's
    dup_amino = np.tile(one_hot_amino, (len(one_hot_dna), 1))
    # return amino acid concatenated with the dna
    return np.concatenate((dup_amino, dup_dna), axis=1)


def groups_b1h_s_score(file_B1H):
    """this function is responsible to separate the b1h_s_score into 8 groups with respect to the suffix """

    group_a1 = file_B1H.filter(regex='A_1$', axis=1)  # suffix A_1
    group_c1 = file_B1H.filter(regex='C_1$', axis=1)  # suffix C_1
    group_g1 = file_B1H.filter(regex='G_1$', axis=1)  # suffix G_1
    group_t1 = file_B1H.filter(regex='T_1$', axis=1)  # suffix T_1
    group_a2 = file_B1H.filter(regex='A_2$', axis=1)  # suffix A_2
    group_c2 = file_B1H.filter(regex='C_2$', axis=1)  # suffix C_2
    group_g2 = file_B1H.filter(regex='G_2$', axis=1)  # suffix G_2
    group_t2 = file_B1H.filter(regex='T_2$', axis=1)  # suffix T_2
    return group_a1, group_c1, group_g1, group_t1, group_a2, group_c2, group_g2, group_t2


def extract_dna_triplet(dna_seq):
    """this function is responsible to extract the first triplet of the dna sequence """
    # in each string take only the first 3 DNA
    i = 0
    for dna in dna_seq:
        dna_seq[i] = dna[0:3]
        i = i + 1
    # find the unique triplets of DNA + sort alphabetically
    dna_seq = list(set(dna_seq))
    dna_seq.sort()
    return dna_seq


def extract_dna_quart(dna_seq):
    """this function is responsible to extract the first triplet of the dna sequence """
    # in each string take only the first 3 DNA
    i = 0
    for dna in dna_seq:
        dna_seq[i] = dna[0:4]
        i = i + 1
    # find the unique triplets of DNA + sort alphabetically
    dna_seq = list(set(dna_seq))
    dna_seq.sort()
    return dna_seq



def main_concatenate_amino_acids_dna_full_amino_seq(file):
    """this function is responsible to to do one hot encoding for the amino acids and the dna.
     the function concatenate the amino acids full sequence with the respective dna. The function returns a matrix where each row consists
     of amino acid and respective dna in one hot encoding"""
    # get data for model + one hot encoding

    oneHot_matrix_amino_acids = ht_one_hot_encode_amino_acids(files_list=file['Amino_Id'])
    oneHot_matrix_DNA = ht_one_hot_encode_DNA(files_list=file['DNA'])
    oneHot_matrix_DNA_vec = []

    # create a matrix: each row represents DNA
    for array in oneHot_matrix_DNA:
        oneHot_matrix_DNA_vec.append(array.flatten())

    oneHot_matrix_amino_acids_vec = []

    # create a matrix: each row represents Amino acid
    for array in oneHot_matrix_amino_acids:
        oneHot_matrix_amino_acids_vec.append(array.flatten())

    # convert list to np
    oneHot_matrix_DNA_vec = np.array(oneHot_matrix_DNA_vec)
    oneHot_matrix_amino_acids_vec = np.array(oneHot_matrix_amino_acids_vec)

    # reshape: each row has: amino acid,DNA
    oneHot_matrix_amino_acids_DNA_vec = np.concatenate((oneHot_matrix_amino_acids_vec, oneHot_matrix_DNA_vec), axis=1)
    return oneHot_matrix_amino_acids_DNA_vec


def main_concatenate_amino_acids_dna_core_amino_seq(file):
    """this function is responsible to to do one hot encoding for the amino acids and the dna.
     the function concatenate the amino acids core sequence with the respective dna. The function returns a matrix where each row consists
     of amino acid core seq and respective dna in one hot encoding"""
    # get data for model + one hot encoding

    oneHot_matrix_amino_acids = ht_one_hot_encode_amino_acids(files_list=file['core_seq'])

    oneHot_matrix_DNA = ht_one_hot_encode_DNA(files_list=file['DNA'])
    oneHot_matrix_DNA_vec = []

    # create a matrix: each row represents DNA
    for array in oneHot_matrix_DNA:
        oneHot_matrix_DNA_vec.append(array.flatten())

    oneHot_matrix_amino_acids_vec = []

    # create a matrix: each row represents Amino acid
    for array in oneHot_matrix_amino_acids:
        oneHot_matrix_amino_acids_vec.append(array.flatten())

    # convert list to np
    oneHot_matrix_DNA_vec = np.array(oneHot_matrix_DNA_vec)
    oneHot_matrix_amino_acids_vec = np.array(oneHot_matrix_amino_acids_vec)

    # reshape: each row has: amino acid,DNA
    oneHot_matrix_amino_acids_DNA_vec = np.concatenate((oneHot_matrix_amino_acids_vec, oneHot_matrix_DNA_vec), axis=1)
    return oneHot_matrix_amino_acids_DNA_vec





def find_dna_triplet(file_all_groups):
    """ this function is responsible to find the dna triplet of each group of the files: file_B1H_A1,...., file_B1H_T2.
     """
    dna_triplet_vec = []
    for file in file_all_groups:
        dna_triplet = list(file.columns.values)
        dna_triplet = dna_triplet[2:]  # choose only relevant columns
        dna_triplet_vec.append(extract_dna_triplet(dna_triplet))
    return dna_triplet_vec


def model_input(file_all_groups, dna_triplet_vec):

    b1h_amino_acid_dna_a1 = b1h_concat(file_all_groups[0], dna_triplet_vec[0])
    b1h_amino_acid_dna_c1 = b1h_concat(file_all_groups[1], dna_triplet_vec[1])
    b1h_amino_acid_dna_g1 = b1h_concat(file_all_groups[2], dna_triplet_vec[2])
    b1h_amino_acid_dna_t1 = b1h_concat(file_all_groups[3], dna_triplet_vec[3])
    b1h_amino_acid_dna_a2 = b1h_concat(file_all_groups[4], dna_triplet_vec[4])
    b1h_amino_acid_dna_c2 = b1h_concat(file_all_groups[5], dna_triplet_vec[5])
    b1h_amino_acid_dna_g2 = b1h_concat(file_all_groups[6], dna_triplet_vec[6])
    b1h_amino_acid_dna_t2 = b1h_concat(file_all_groups[7], dna_triplet_vec[7])

    return [b1h_amino_acid_dna_a1, b1h_amino_acid_dna_c1, b1h_amino_acid_dna_g1, b1h_amino_acid_dna_t1,
            b1h_amino_acid_dna_a2, b1h_amino_acid_dna_c2, b1h_amino_acid_dna_g2,b1h_amino_acid_dna_t2]


def b1h_concat(file, dna_triplet):

    # one hot encoding vector for the core seq and the DNA
    oneHot_B1H_DNA = oneHot_DNA_vec(dna_triplet)
    oneHot_B1H_amino = oneHot_Amino_acid_vec(file['core_seq'])

    # concatenate amino acid with dna and create a matrix where each row is one amino acid with DNA
    return concat(oneHot_B1H_amino, oneHot_B1H_DNA)


def create_label(file_all_groups):
    # y_a1: labels
    y_a1 = file_all_groups[0].filter(regex='A_1$', axis=1)
    y_a1 = y_a1.values
    y_a1 = y_a1.T.flatten()

    # y_c1: labels
    y_c1 = file_all_groups[1].filter(regex='C_1$', axis=1)
    y_c1 = y_c1.values
    y_c1 = y_c1.T.flatten()

    # y_g1: labels
    y_g1 = file_all_groups[2].filter(regex='G_1$', axis=1)
    y_g1 = y_g1.values
    y_g1 = y_g1.T.flatten()

    # y_t1: labels
    y_t1 = file_all_groups[3].filter(regex='T_1$', axis=1)
    y_t1 = y_t1.values
    y_t1 = y_t1.T.flatten()

    # y_a2: labels
    y_a2 = file_all_groups[4].filter(regex='A_2$', axis=1)
    y_a2 = y_a2.values
    y_a2 = y_a2.T.flatten()

    # y_c2: labels
    y_c2 = file_all_groups[5].filter(regex='C_2$', axis=1)
    y_c2 = y_c2.values
    y_c2 = y_c2.T.flatten()

    # y_g2: labels
    y_g2 = file_all_groups[6].filter(regex='G_2$', axis=1)
    y_g2 = y_g2.values
    y_g2 = y_g2.T.flatten()

    # y_t2: labels
    y_t2 = file_all_groups[7].filter(regex='T_2$', axis=1)
    y_t2 = y_t2.values
    y_t2 = y_t2.T.flatten()

    return [y_a1, y_c1, y_g1, y_t1, y_a2, y_c2, y_g2, y_t2]


def find_groups_4_dna(file, dna_quart):
    groups = file['groups']
    # duplicate groups as the number of dna
    groups = np.tile(groups, (len(dna_quart), 1))
    return groups.flatten()









































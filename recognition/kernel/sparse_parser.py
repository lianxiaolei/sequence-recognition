# coding:utf8

import numpy as np

# DIGITS = '0123456789'
characters = '0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20'
char_list = characters.split('_')


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: A list of lists of type dtype where each element is a sequence
        dtype: Type of SparseTensor
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        # print(n, len(seq))
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        # str = int(DIGITS[spars_tensor[1][m]])
        str = int(char_list[spars_tensor[1][m]])
        decoded.append(str)
    return decoded


if __name__ == '__main__':
    sequence = [[0, 0], [0, 1, 1, 4], [1, 0, 1]]
    # label = [[1, 1], 2, 3]
    a, b, c = sparse_tuple_from(sequence)
    print(a)
    print('--' * 20)
    print(b)
    print('--' * 20)
    print(c)
    print(decode_sparse_tensor((a, b, c)))

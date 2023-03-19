# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter
from sklearn.utils import shuffle
from itertools import chain


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.

    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # get the total number of entries
    num_entries = len(seqs)
    # the balance class size is the total number of entries divided by the number of classes for equal split
    # only 2 classes: positive and negative
    balanced_size = int(np.ceil(num_entries / 2))

    # get the count of class frequencies
    label_counts = Counter(labels).values()
    # get the less frequent class name
    less_frequent_label = list(Counter(labels).keys())[1]
    # get pos class indices
    pos_indices = [i for i, x in enumerate(labels) if x == True]
    # get neg class
    neg_indices = [i for i, x in enumerate(labels) if x == False]
    # resample using random indices
    # get random indices from total positive and negative indices lists
    pos_sample_indices = np.random.choice(pos_indices, size=balanced_size, replace=True)
    neg_sample_indices = np.random.choice(neg_indices, size=balanced_size, replace=True)

    # get sequences from the random indices
    sampled_seqs = [seqs[i] for i in pos_sample_indices] + [seqs[i] for i in neg_sample_indices]
    # get labels from the random indices
    sampled_labels = [labels[i] for i in pos_sample_indices] + [labels[i] for i in neg_sample_indices]
    # shuffle the final sequences and labels in the same manner to keep position consistent
    sampled_seqs, sampled_labels = shuffle(sampled_seqs, sampled_labels)

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encodings = []

    # iterate through nucleotides
    # append to final encodings list the new one-hot encoded values
    for seq in seq_arr:
        seq_encodings = []
        for nuc in seq:
            if nuc == 'A':
                seq_encodings.append(np.array([1, 0, 0, 0]))
            elif nuc == 'T':
                seq_encodings.append(np.array([0, 1, 0, 0]))
            elif nuc == 'C':
                seq_encodings.append(np.array([0, 0, 1, 0]))
            elif nuc == 'G':
                seq_encodings.append(np.array([0, 0, 0, 1]))

        sequence_encoded = list(chain.from_iterable(seq_encodings))
        encodings.append(sequence_encoded)

    # return final encodings as an array
    return np.array(encodings, dtype=object)
    #return list(chain.from_iterable(np.array(encodings)))
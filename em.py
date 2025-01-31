from utils import process_file, split_to_n, filter_rare_words
from itertools import chain
import numpy as np


def initialize_EM(articles):
    vocabulary = set(chain.from_iterable(articles))
    num_groups = 9
    groups, labels = split_to_n(articles, num_groups)
    num_docs = labels.shape[0]
    alpha_i, bins = np.histogram(labels, num_groups, density=False)
    alpha_i = alpha_i / float(num_docs)

    return alpha_i,
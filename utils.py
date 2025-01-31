# Eliyahu Dagi
# 036781839
from language_models import LanguageModel, UniGramModel, LindstoneSmoother
import math
from typing import List
from itertools import chain
from typing import List
import numpy as np


def flatten_lists(l: List[List]) -> List:
    return list(chain.from_iterable(l))


def perplexity(model: LanguageModel, events):
    """

    :param model: some language model which can give p(word)
    :param events: some datasets of words
    :return: perplexity value
    """
    probes = [model.p(event) for event in events]
    if 0 in probes:
        return 0
    # fixed use log2
    pm = sum(map(math.log2, probes))
    res = math.pow(2, pm * (-1. / len(events)))
    return res


def process_file(path):
    """
    read file and parse it to list of words according to the format described in the exercise
    :param path: path to the txt file
    :return: list of words
    """
    with open(path, 'r') as f:
        raw_data = f.read()
    lines = raw_data.split('\n')
    articles = [lines[i] for i in range(2, len(lines), 4)]
    articles_words = [[word for word in article.split(' ') if word] for article in articles]
    return articles_words


def split_to_n(elements, n_groups):
    groups = [[] for _ in range(n_groups)]
    labels = []
    for i, element in enumerate(elements):
        groups[i % n_groups].append(element)
        labels.append(i % n_groups)
    return groups, np.array(labels)


def split_dataset(dataset: List[str], ratio: float):
    """

    :param dataset: dataset to split
    :param ratio: ratio of the first part (ratio * len(dataset)) the second will be 1 - ratio size
    :return: 2 datasets
    """
    size_train = int(round(ratio * len(dataset)))
    return dataset[:size_train], dataset[size_train:]


def filter_rare_words(articles: List[List[str]], min_appearance: int):
    all_words = flatten_lists(articles)
    rare_words = UniGramModel(all_words).get_rare_events(min_appearance + 1)
    articles = [list(filter(lambda x: x not in rare_words, article))for article in articles]
    return articles


# fixed - change the function to choose max instead of min
def calibrate_lindstone(model: LindstoneSmoother, val_dataset, start, end, step = 0.01):
    max_perplexity = -999999
    max_alpha = -1
    for alpha_val in [start + i * step for i in range(int((end - start) / step))]:
        model.set_alpha(alpha_val)
        cur_perplexity = perplexity(model, val_dataset)
        if cur_perplexity > max_perplexity:
            max_perplexity = cur_perplexity
            max_alpha = alpha_val
    return max_alpha, max_perplexity


def ensure_model_sum(model: UniGramModel, dataset, unseen_word, n_unseen, eps=1e-5):
    p_unseen = model.p(unseen_word)
    sum_p = 0
    for event in set(dataset):
        if model.get_count(event) > 0:
            sum_p += model.p(event)

    sum_p += n_unseen * p_unseen
    assert abs(sum_p - 1) < eps


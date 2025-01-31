# Eliyahu Dagi
# 036781839
from collections import Counter
from abc import ABC
from typing import Set


class LanguageModel(ABC):
    def p(self, event) -> float:
        pass


class UniGramModel(LanguageModel):
    """
    UniGramModel - p(word) = count word appearance / total words in dataset
    """
    def __init__(self, dataset):
        self.counter = Counter(dataset)
        self.sum = float(sum(self.counter.values()))

    def p(self, event):
        return self.p_by_count(self.get_count(event))

    def p_by_count(self, event_count):
        return event_count / self.sum

    def get_count(self, event):
        if event in self.counter:
            return self.counter[event]
        else:
            return 0

    def get_rare_events(self, min_appearence) -> Set[str]:
        rare_events = set()
        for event, count in self.counter.items():
            if count < min_appearence:
                rare_events.add(event)
        return rare_events

    def __len__(self):
        return len(self.counter.keys())


class LindstoneSmoother(UniGramModel):
    """
        LindstoneSmoother - p(w) = c(word) + alpha / (dataset_len + alpha * num_different_words)
        """
    def __init__(self, dataset, alpha):
        super().__init__(dataset)
        self.alpha = -1
        self.denominator_add = -1
        self.set_alpha(alpha)

    def p(self, event):
        count = self.get_count(event)
        return self.p_by_count(count)

    def p_by_count(self, event_count):
        return (event_count + self.alpha) / (self.sum + self.denominator_add)

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.denominator_add = len(self.counter.keys()) * self.alpha


class HeldOut(UniGramModel):
    def __init__(self, t_set, h_set):
        super().__init__(t_set)
        self.count_to_prob, self.count_NrTr = self.calc_count_to_probe(h_set)


    def p(self, event):
        """

        :param event: word
        :return: probability find the matching count and use it to get pre calculated probability
        """
        count = self.get_count(event)
        return self.p_by_count(count)

    def p_by_count(self, event_count):
        return self.count_to_prob[event_count]

    def group_by_counts(self):
        """

        :return: Dict where each key is r(number of appearences) and the value is a set of all words which c(word) = r
        """
        count_groups = dict()
        for word, count in self.counter.items():
            if count not in count_groups:
                # initialize set in first occurrence
                count_groups[count] = set()
            count_groups[count].add(word)
        return count_groups

    def calc_count_to_probe(self, h_set):
        """

        :param h_set: the hold out set
        :return: count_to_probe - mapping between count( number of appearences in train set to probability
                and count_NrTr - additional info used for calculating the probabilty for each count number of different
                words in train set(Nr) whith the sam count and total events in hold out set(Tr)
        """
        # calculate for each count : all words where c(word) == count
        words_by_count = self.group_by_counts()
        # add missing words
        missing_words = set()
        for word in h_set:
            if self.get_count(word) == 0:
                missing_words.add(word)
        words_by_count[0] = missing_words
        h_model = UniGramModel(h_set)
        h_len = float(len(h_set))
        count_to_probe = dict()
        count_NrTr = dict()
        words_by_count = dict(sorted(words_by_count.items(), key=lambda item: item[0]))
        for count, words in words_by_count.items():
            Nr = len(words)
            Tr = sum([h_model.get_count(word) for word in words])
            count_to_probe[count] = Tr / (Nr * h_len)
            count_NrTr[count] = (Nr, Tr)
        count_NrTr = dict(sorted(count_NrTr.items(), key=lambda item: item[0]))
        return count_to_probe, count_NrTr

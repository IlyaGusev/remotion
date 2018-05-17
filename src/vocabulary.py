# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Vocabulary with save/load support.

from collections import Counter
from typing import List, Tuple
import pickle
import logging


class Vocabulary:
    def __init__(self):
        self.index2word = list()
        self.word2index = dict()
        self.word2count = Counter()
        self.reset()

    def reset(self):
        self.word2count = Counter()
        self.index2word = ["<pad>", "<unk>"]
        self.word2index = {word: index for index, word in enumerate(self.index2word)}

    def get_pad(self):
        return self.word2index["<pad>"]

    def get_unk(self):
        return self.word2index["<unk>"]

    def add_sentence(self, sentence):
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.word2count[word] += 1
            self.index2word.append(word)
        else:
            self.word2count[word] += 1

    def has_word(self, word) -> bool:
        return word in self.word2index

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.get_unk()

    def get_word(self, index):
        return self.index2word[index]

    def size(self):
        return len(self.index2word)

    def is_empty(self):
        empty_size = 2
        return self.size() <= empty_size

    def shrink(self, n):
        best_words = self.word2count.most_common(n)
        self.reset()
        for word, count in best_words:
            self.add_word(word)
            self.word2count[word] = count

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

    def get_indices(self, sentence: str) -> List[int]:
        return [self.get_index(word) for word in sentence.strip().split()]

    def pad_indices(self, indices: List[int], max_length: int):
        return indices + [self.get_pad() for _ in range(max_length - len(indices))]

    def merge(self, other):
        for word in other.index2word:
            self.add_word(word)
        return self

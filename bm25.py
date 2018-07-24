#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from six.moves import xrange


# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):

    def __init__(self, corpus, delta=0):
        self.corpus_size = len(corpus)
        self.dl_length = [sum(x.itervalues()) for x in corpus]
        self.avgdl = sum(self.dl_length) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.delta = delta
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += document[word]
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        self.average_idf = sum(float(val) for val in self.idf.values()) / len(self.idf)
        return

    def get_score(self, document, index):
        score = 0
        average_idf = self.average_idf
        doc_size_ratio = float(self.dl_length[index]) / self.avgdl
        delta = self.delta
        for word, tf in document.iteritems():
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * tf) * ((PARAM_K1 + 1) / (1 + PARAM_K1 * (1 - PARAM_B + PARAM_B * doc_size_ratio) / self.f[index][word]) +  delta)
        return score

    def get_scores(self, document):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)
    return weights

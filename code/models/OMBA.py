import numpy as np
import random
import math
from collections import defaultdict
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity
import itertools


def cosine(list1, list2):
    return cosine_similarity([list1], [list2])[0][0]


class OMBA:
    def __init__(self, pd, buffered_relations=None, locations=None):
        self.pd = pd
        self.nt2vecs = defaultdict(lambda: defaultdict(
            lambda: (np.random.rand(pd['dim'])-0.5)/pd['dim']))
        self.grad_nt2vecs = defaultdict(lambda:
                                        defaultdict(lambda:
                                                    np.zeros(pd['dim'])))

        self.nt2negas = defaultdict(list)
        self.buffer = []
        self.regu_weight = pd['regu_weight']

    def intra_agreement(self, relation, type='avg'):
        nt2vecs = self.nt2vecs
        ws_vec = (np.random.rand(self.pd['dim'])-0.5)/self.pd['dim']
        us_vec = (np.random.rand(self.pd['dim'])-0.5)/self.pd['dim']

        if 'w' in relation:
            w_vecs = [nt2vecs['w'][w] for w in relation['w']]

            if type == 'avg':
                ws_vec = np.average(w_vecs, axis=0) if w_vecs\
                    else np.zeros(self.pd['dim'])
                w_vecs = [ws_vec]

        if 'u' in relation:
            for u in relation['u']:
                us_vec = nt2vecs['u'][u]

        vecs = [us_vec] + w_vecs
        scores = [expit(np.dot(vec1, vec2))
                  for vec1, vec2 in itertools.combinations(vecs, r=2)]
        score = sum(scores)/(len(scores) + np.finfo(float).eps)
        return score

    def partial_fit(self, relations):
        pd = self.pd
        nt2vecs = self.nt2vecs
        self.alpha = pd['alpha']
        sample_size, sampled_size = len(relations)*pd['epoch'], 0

        self.buffer = []
        self.buffer += relations
        relations = self.buffer

        # if no samples could be found in the bin
        if sample_size == 0:
            return nt2vecs

        while True:
            random.shuffle(relations)
            for relation in relations:
                ada_weight = (1/math.e**(pd['regu_weight'] *
                                         self.intra_agreement(relation)))

                if self.alpha > pd['alpha'] * 1e-4:
                    self.alpha -= pd['alpha'] * 1e-6
                sum_vec = np.zeros(pd['dim'])
                sum_weight = 0

                for nt in relation:
                    for n in relation[nt]:
                        self.nt2negas[nt].append(n)
                        sum_vec += nt2vecs[nt][n] * relation[nt][n]
                        sum_weight += relation[nt][n]

                for nt in relation:
                    for n in relation[nt]:
                        for j in range(pd['negative']+1):
                            minus_n_vec = sum_vec - nt2vecs[nt][n]\
                                * relation[nt][n]

                            minus_n_weight = sum_weight - relation[nt][n]
                            minus_n_vec_avg = minus_n_vec / minus_n_weight

                            if j == 0:
                                target = n
                                label = 1
                            else:
                                target = random.choice(self.nt2negas[nt])
                                if target == n:
                                    continue
                                label = 0
                            f = np.dot(nt2vecs[nt][target], minus_n_vec_avg)
                            g = (label - expit(f))

                            for nt2 in relation:
                                for n2 in relation[nt2]:
                                    if not (nt == nt2 and n == n2):
                                        grad = -g*nt2vecs[nt][target] * \
                                            relation[nt2][n2]/minus_n_weight
                                        self.grad_nt2vecs[nt2][n2] += grad**2
                                        update = -self.alpha*grad*ada_weight
                                        update /= np.sqrt(self.grad_nt2vecs
                                                          [nt2][n2] +
                                                          np.finfo(float).eps)
                                        nt2vecs[nt2][n2] += update
                            grad = -g*minus_n_vec_avg
                            self.grad_nt2vecs[nt][target] += grad**2
                            update = -self.alpha*grad*ada_weight
                            update /= np.sqrt(self.grad_nt2vecs[nt]
                                              [target] + np.finfo(float).eps)
                            nt2vecs[nt][target] += update
                        sampled_size += 1
                        if sampled_size == sample_size:
                            return nt2vecs

import numpy as np
from collections import defaultdict
from time import time as cur_time
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit

from models.OMBA import OMBA


def cosine(list1, list2):
    return cosine_similarity([list1], [list2])[0][0]


def convert_ts(ts):
    return (ts % 3600) % 7


class EmbedPredictor(object):
    def __init__(self, pd):
        self.pd = pd
        self.nt2vecs = defaultdict(lambda: defaultdict(
            lambda: (np.random.rand(pd['dim'])-0.5)/pd['dim']))
        self.start_time = cur_time()
        self.product_counts = defaultdict(lambda: 0)
        self.emb_size = pd['dim']
        self.embed_algo = OMBA(self.pd)

    def partial_fit(self, tweets):
        relations = self.prepare_training_data(tweets)
        self.nt2vecs = self.embed_algo.partial_fit(relations)

    def prepare_training_data(self, tweets):
        relations = []
        for tweet in tweets:
            nts = self.pd['nt_list'][1:]

            t = convert_ts(tweet.ts)
            u = tweet.uid
            words = tweet.words

            relation = defaultdict(lambda: defaultdict(float))
            for id, w in enumerate(words):
                val = tweet.values[id]
                relation['w'][w] += (min(val, 10)**(2.3))/1.3
                self.product_counts[w] += 1

            weight_sum_w = sum(relation['w'].values())
            if weight_sum_w == 0:
                weight_sum_w = 1

            for nt in nts:
                relation[nt][eval(nt)] += weight_sum_w

            # remove transactions with one product
            if len(relation['w']) == 1:
                continue
            relations.append(relation)

        return relations

    def gen_user_feature(self, user, predict_type='w'):
        nt2vecs = self.nt2vecs
        us_vec = nt2vecs['u'][user] if user in nt2vecs['u']\
            else np.zeros(self.emb_size)
        return us_vec

    def gen_temporal_feature(self, time, predict_type='w'):
        nt2vecs = self.nt2vecs
        t = convert_ts(time)
        ts_vec = nt2vecs['t'][t] if t in nt2vecs['t']\
            else np.zeros(self.emb_size)
        return ts_vec

    def gen_textual_feature(self, words, predict_type='w'):
        nt2vecs = self.nt2vecs

        ws_vec = np.zeros(self.emb_size)
        w_vecs = [nt2vecs['w'][w] for w in words if w in nt2vecs['w']]

        ws_vec = np.average(w_vecs, axis=0) if w_vecs\
            else np.zeros(self.emb_size)
        return ws_vec

    def normalize_vec(self, vec):
        return vec / (np.linalg.norm(vec) + np.finfo(float).eps)

    def predict(self, time, words, user, target, predict_type='w'):
        w_vec = self.gen_textual_feature(words, predict_type=predict_type)
        t_vec = self.gen_textual_feature([target], predict_type=predict_type)
        vecs = [w_vec, t_vec]

        if 't' in self.pd['nt_list']:
            vecs.append(self.gen_temporal_feature(time, predict_type))

        if 'u' in self.pd['nt_list']:
            vecs.append(self.gen_user_feature(user, predict_type))

        for i in range(len(vecs)):
            vecs[i] = self.normalize_vec(vecs[i])
        score = sum([cosine(vec1, vec2)
                     for vec1, vec2 in itertools.combinations(vecs, r=2)])
        return round(score, 6)

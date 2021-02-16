import bisect
import numpy as np
from copy import deepcopy
# from embed import *


class QuantitativeEvaluator:
    def __init__(self, predict_type='w', fake_num=10):
        self.ranks = []
        self.target_counts = []
        self.predict_type = predict_type
        self.fake_num = fake_num

    def get_ranks(self, tweets, predictor):
        rand_seed = 1
        np.random.seed(rand_seed)
        noiseList = np.random.choice(tweets,
                                     self.fake_num*len(tweets)).tolist()

        for tweet in tweets:
            noise_tweets = []
            for i in range(self.fake_num):
                noise_tweets.append(noiseList.pop())

            for w in tweet.words:
                self.target_counts.append(predictor.product_counts[w])
                neig_words = deepcopy(tweet.words)
                neig_words.remove(w)
                scores = []
                score = predictor.predict(tweet.ts, neig_words, tweet.uid, w,
                                          predict_type=self.predict_type)
                scores.append(score)

                for n_t in noise_tweets:
                    temp_n_t = deepcopy(n_t.words)

                    for n_w in temp_n_t:
                        if n_w not in tweet.words:
                            noise_score = predictor.predict(
                                tweet.ts, neig_words, tweet.uid, n_w,
                                predict_type=self.predict_type)
                            scores.append(noise_score)
                            break
                scores.sort()

                # handle ties
                rank = len(scores)+1-(bisect.bisect_left(scores, score) +
                                      bisect.bisect_right(scores, score)+1)/2.0
                self.ranks.append(rank)
        mrr, mr = self.compute_mrr()

    def compute_mrr(self):
        ranks = self.ranks
        rranks = [1.0/rank for rank in ranks]
        mrr, mr = sum(rranks)/len(rranks), sum(ranks)/len(ranks)
        return round(mrr, 4), round(mr, 4)

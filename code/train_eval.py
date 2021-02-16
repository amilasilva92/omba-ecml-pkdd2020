from paras import load_params
from dataset import read_tweets
from dataset import get_voca, update_tweets
from collections import defaultdict
from evaluator import QuantitativeEvaluator
from embed import EmbedPredictor
import os
import psutil
import time
import random
import numpy as np
import sys


def set_rand_seed(pd):
    rand_seed = pd['rand_seed']
    np.random.seed(rand_seed)
    random.seed(rand_seed)


def read_data(pd):
    start_time = time.time()
    tweets = read_tweets(pd['tweet_file'])

    tweets.sort(key=lambda tweet: tweet.ts)
    voca = get_voca(tweets, pd['voca_min'], pd['voca_max'])

    if pd['update_tweets']:
        tweets = update_tweets(tweets, voca)
    print('Reading data done, elapsed time: ',
          round(time.time()-start_time))

    print('Total number of tweets: ', len(tweets))
    return tweets, voca


def train_and_evaluate(tweets, voca, model_type='embed'):
    # type = ['embed', 'nmf', 'count', 'prod2vec', 'prod2vec_o']
    print('#########################')
    print('Model Type: ', model_type)
    print('#########################')
    start_time = time.time()
    evaluators = [QuantitativeEvaluator(predict_type=predict_type, fake_num=10)
                  for predict_type in pd['predict_type']]
    day2batch = defaultdict(list)
    for tweet in tweets:
        day = tweet.ts/3600
        day2batch[day].append(tweet)

    batches = [day2batch[d] for d in sorted(day2batch)]
    test_batch_indices = np.random.choice(
        range(len(batches)/2, len(batches)),
        pd['test_batch_num'], replace=False)

    model = EmbedPredictor(pd)
    print('#########################')
    print('Count Measure: ', pd['update_strategy'])
    print('#########################')

    # training_batch = []
    for i, batch in enumerate(batches):
        if i % 200 == 0:
            print('time:', time.time()-start_time)

        if i in test_batch_indices:
            print('results for batch', i)
            for evaluator in evaluators:
                evaluator.get_ranks(batch, model)
                mrr, mr = evaluator.compute_mrr()
                print(evaluator.predict_type, 'mr:', mr, 'mrr:', mrr)
        model.partial_fit(batch)

    for evaluator in evaluators:
        mrr, mr = evaluator.compute_mrr()
        print(evaluator.predict_type, 'mr:', mr, 'mrr:', mrr)

    print('Model training and evaluation done, elapsed time: ',
          round(time.time()-start_time))

    return model


def run(pd):
    set_rand_seed(pd)
    tweets, voca = read_data(pd)
    model = train_and_evaluate(tweets, voca, 'embed')
    del model


def dump_tweets(tweets, file_path=None):
    out_file = open(file_path, 'w')
    for tweet in tweets:
        out_file.write(str(tweet))


def write_embeddings(model, pd):
    directory = pd['model_embeddings_dir']
    if not os.path.isdir(directory):
        os.makedirs(directory)

    type = '_'.join(pd['nt_list'])
    for nt, vecs in model.nt2vecs.items():
        with open(directory+nt + '_' + type + '_user.txt', 'w') as f:
            for node, vec in vecs.items():
                line = [str(e) for e in [node, list(vec)]]
                f.write('\x01'.join(line)+'\n')


if __name__ == '__main__':
    pid = os.getpid()
    ps = psutil.Process(pid)

    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    run(pd)

    memoryUse = ps.memory_info()
    print('\n============================')
    print('Info about the memory usage,')
    print('============================')
    print('rss(MB):', memoryUse.rss/1048576.0)
    print('vms(MB):', memoryUse.vms/1048576.0)
    print('============================')

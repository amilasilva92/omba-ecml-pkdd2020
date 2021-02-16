from collections import defaultdict
from transaction_handler import Tweet


def read_tweets(tweet_file):
    tweets = []
    for line in open(tweet_file):
        tweet = Tweet()
        tweet.load_tweet(line.strip())
        tweets.append(tweet)
    return tweets


def get_voca(tweets, voca_min=0, voca_max=20000):
    word2freq = defaultdict(int)
    for tweet in tweets:
        for word in tweet.words:
            word2freq[word] += 1
    word_and_freq = word2freq.items()
    word_and_freq.sort(reverse=True, key=lambda tup: tup[1])
    voca = set(zip(*word_and_freq[voca_min:voca_max])[0])
    if '' in voca:
        voca.remove('')
    return voca


def update_tweets(tweets, voca):
    for tweet in tweets:
        temp_words = []
        for w in tweet.words:
            if w in voca:
                temp_words.append(w)
        tweet.words = temp_words
    return tweets

from nltk import *
from nltk.corpus import stopwords
def max_frequent_bigrams(words):
    stopwords_set = set(stopwords.words(fileids = u'english'))
    bigram_words = bigrams([w.lower() for w in words])
    filtered_bigram_words = [bigram_word for bigram_word in bigram_words if bigram_word[0] not in stopwords_set and bigram_word[1] not in stopwords_set]
    fdist = FreqDist(filtered_bigram_words)
    sorted_filtered_bigram_words = sorted(fdist.keys(), key = lambda x:fdist[x], reverse = True)
    return sorted_filtered_bigram_words[:50:]
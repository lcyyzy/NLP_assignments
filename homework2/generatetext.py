import nltk
from nltk import *
import random
from random import choice

def generate_model(word, num = 15):
    text = nltk.corpus.genesis.words('english-kjv.txt')
    for i in range(num):
        bigrams = nltk.bigrams(text)
        print(word),
        link_words = []
        for bigram in bigrams:
            if bigram[0] == word:
                link_words.append(bigram[1])
        word = random.choice(link_words)

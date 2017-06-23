import nltk
from nltk.corpus import brown

def performance(size):
    brown_tagged_sents = brown.tagged_sents(categories = 'news')
    train_tagged_sents = brown_tagged_sents[:size]
    unigram_tagger = nltk.UnigramTagger(train_tagged_sents)
    return unigram_tagger.evaluate(brown_tagged_sents)

def display():
    import pylab
    sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    perfs = []
    for s in sizes:
        perfs.append(performance(s))
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Unigram Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

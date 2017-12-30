'''
Please install package as below, before start this code!
pip install textblob
pip install -U textblob nltk
python -m textblob.download_corpora
http://textblob.readthedocs.io/en/dev/classifiers.html
'''

from textblob.classifiers import NaiveBayesClassifier
# from textblob import TextBlob

train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]
test = [
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

clf = NaiveBayesClassifier(train)
print("test result:", clf.accuracy(test))

# Classify some text
print(clf.classify("Their burgers are amazing."))  # "pos"
print(clf.classify("I don't like their pizza."))   # "neg"
print(clf.classify("This is amazing library!"))    # "Pos"
print(clf.classify("But the hangover is horrible")) #"Neg"

new_data = [('She is my best friend.', 'pos'),
            ("I'm happy to have a new friend.", 'pos'),
            ("Stay thirsty, my friend.", 'pos'),
            ("He ain't from around here.", 'neg')]

clf.update(new_data)
print(clf.accuracy(test))


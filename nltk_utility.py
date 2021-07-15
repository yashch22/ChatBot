import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
nltk.download('punkt')

stemmer = PorterStemmer()
# Tokenizing a sentence.
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Convert word to stem word
def stem(word):
    return stemmer.stem(word.lower())


# Converting sentence to a vector.

def bag_of_words(tokenized_sentence, vocabulary):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(vocabulary),dtype=np.float32)

    for idx, w in enumerate(vocabulary):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag



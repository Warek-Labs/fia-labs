import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def lower(text: list[str]) -> list[str]:
    return [w.lower() for w in text]


def stem(word: str | list[str]) -> str | list[str]:
    if isinstance(word, list):
        return [stemmer.stem(w) for w in word]
    return stemmer.stem(word)


def bag_of_words(tokenized_sentence: list[str], all_words: list[str]) -> np.ndarray[int, np.dtype]:
    tokenized_sentence = stem(tokenized_sentence)
    bag = np.zeros(len(all_words), dtype=int)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag


if __name__ == '__main__':
    pass

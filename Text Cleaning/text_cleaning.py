import string
import re
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download("words")
nltk.download("stopwords")
nltk.download("wordnet")


class TextCleaning:
    def __init__(self, stemming: bool):
        self._ps = PorterStemmer()

        if type(stemming) is not bool:
            raise TypeError("Parameter < stemming > must be < bool >")
        else:
            self._stem = stemming

        self._double = pd.read_csv(
            "https://raw.githubusercontent.com/NITHISHM2410/Text_Preprocessing/NLP/Text%20Cleaning/utils/double_identical.txt",
            header=None).values.reshape((-1,)).tolist()

        self._notwords = self.ready_not_words()
        self._sw = self._imp_stop_words()

    def ready_not_words(self):
        words = pd.read_csv(
            "https://raw.githubusercontent.com/NITHISHM2410/Text_Preprocessing/NLP/Text%20Cleaning/utils/negative%20words.txt",
            header=None).values.reshape((-1,)).tolist()
        words.append("not")
        return words

    def _imp_stop_words(self,):
        stop_words = stopwords.words("english")
        stop_words = set(stop_words) - set(self._notwords)
        stop_words = [i.replace("'", "") for i in stop_words]

        return stop_words

    def _duplicates(self, text):
        output = ""
        prev = ""
        for t in text:
            if len(output) == 0:
                output += t
                prev = t
            if t == prev:
                continue
            else:
                output += t
                prev = t
        return output

    def clean(self, sent):
        sent = sent.lower()
        sent = re.sub(r'[^\w\s]', '', sent)
        sent = re.sub(r'\d+', '', sent)
        sent = sent.split()

        for i in sent:
            if 'http' in i or 'html' in i or 'www' in i or '.com' in i:
                sent.remove(i)
        sent = [i for i in sent if not len(i) <= 2]

        for i in sent:
            ind = sent.index(i)
            if i in self._double:
                continue
            else:
                i = self._duplicates(i)
                sent[ind] = i

        sent = [i for i in sent if i not in self._sw]

        if self._stem:
            sent = [self._ps.stem(i) for i in sent]
        sent = " ".join(sent)

        return sent

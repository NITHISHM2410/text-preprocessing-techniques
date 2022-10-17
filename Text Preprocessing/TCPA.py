import nltk
import pandas as pd
import numpy as np
import enchant
import string
from nltk.stem import PorterStemmer
nltk.download("words")
from nltk.corpus import words as eng_words
from nltk.corpus import stopwords

class TextCleaning:
   def __init__(self,stemming: bool):
       self._ps = PorterStemmer()
       if type(stemming) is not bool:
        raise TypeError("Parameter < stemming > must be < bool >")
       else:
        self._stem = stemming
       self._eng = enchant.Dict("en_US")
       self._sw = self._imp_stop_words()
       self._double = np.squeeze(pd.read_csv(
           "https://raw.githubusercontent.com/NITHISHM2410/Text_Cleaner/main/double_identical.txt?token=GHSAT0AAAAAABSDA6PT5NO7HGVC7VPTC3GMYRAFW6A",
           header=None).values).tolist()
       self._notwords = np.squeeze(pd.read_csv("https://raw.githubusercontent.com/NITHISHM2410/Text_Cleaner/main/negative%20words.txt",
                                               header=None).values.tolist())
       self._notwords = [i.replace("'", "") for i in self._notwords]


   @staticmethod
   def _imp_stop_words():
       words = stopwords.words("english")
       words.append("hello")
       ntwords = [i for i in words if "n't" in i or i[-1] == 'n']
       ntwords.remove('between')
       ntwords.remove('again')
       ntwords.remove('on')
       ntwords.remove('an')
       ntwords.remove('won')
       ntwords.remove('when')
       ntwords.remove('than')
       ntwords.remove('then')
       ntwords = ntwords[5:]
       stop_words = set(words) - set(ntwords)
       return stop_words

   @staticmethod
   def _remove_consec_duplicates(s):
       new_s = ""
       prev = ""
       for c in s:
           if len(new_s) == 0:
               new_s += c
               prev = c
           if c == prev:
               continue
           else:
               new_s += c
               prev = c
       return new_s

   def text_preprocessing(self,series):
       series = series.apply(self._cleaner)
       return series

   def _cleaner(self,sent):
       sent = sent.split()
       sent = [i.lower() for i in sent]
       sent = [i for i in sent if i not in self._sw]
       for i in sent:
           if 'http' in i or 'html' in i or 'www' in i or '.com' in i:
               sent.remove(i)
       sent = " ".join(sent)
       sent = list(sent)
       sent = [i for i in sent if not i.isdigit()]
       sent = "".join(sent)
       sent = sent.split()
       sent = " ".join(sent)
       sent = list(sent)
       sent = "".join([i for i in sent if i not in string.punctuation])
       sent = sent.split()
       sent = [i for i in sent if not len(i) <= 2]
       for i in sent:
           ind = sent.index(i)
           if i in self._double:
               continue
           else:
               i = self._remove_consec_duplicates(i)
               sent[ind] = i
       for i in sent:
           if i in self._notwords:
               continue
           elif i not in eng_words.words():
               sent.remove(i)
       if self._stem:
           sent = [self._ps.stem(i) for i in sent]
       sent = " ".join(sent)

       return sent



      
      

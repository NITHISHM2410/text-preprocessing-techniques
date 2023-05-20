class TextCleaning:
    def __init__(self, stemming: bool):
        self._ps = PorterStemmer()
        if type(stemming) is not bool:
            raise TypeError("Parameter < stemming > must be < bool >")
        else:
            self._stem = stemming
        self._eng = enchant.Dict("en_US")
        self._sw = self._imp_stop_words()
        self._double = np.squeeze(pd.read_csv(
            "https://raw.githubusercontent.com/NITHISHM2410/Text_Preprocessing/NLP/Text%20Cleaning/double_identical.txt",
            header=None).values).tolist()
        self._notwords = np.squeeze(pd.read_csv(
            "https://raw.githubusercontent.com/NITHISHM2410/Text_Preprocessing/NLP/Text%20Cleaning/negative%20words.txt",
            header=None).values.tolist())
        self._notwords = self.ready_not_words()
        self._eng_words = self.ready_eng_words()

    def ready_eng_words(self):
        self._eng_words = eng_words.words()
        self._eng_words += self._notwords
        return self._eng_words

    def ready_not_words(self):
        self._notwords = [i.replace("'", "") for i in self._notwords]
        self._notwords.append("not")
        return self._notwords

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
        stop_words.remove("not")
        return stop_words

    def not_list(self):
        return self._notwords

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


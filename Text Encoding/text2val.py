class text2val():
    def __init__(self, char_maxlen, word_maxlen, char_padding_type='post', char_trunc_type='post',
                 word_padding_type='post', word_trunc_type='post',
                 word_pad_value='-', char_pad_value=0):
        """
        Parameters:
           char_maxlen : maximum number of characters in a word to be considered while encoding

           word_maxlen : maximum number of words in a sentence to be considered while encoding

           char_padding_type : type of padding to be done on a word when number of characters is less than < char_maxlen >

           word_padding_type : type of padding to be done on a sentence when number of words is less than < word_maxlen >

           char_trunc_type : type of truncatiom to be done on a word when number of characters exceeds < char_maxlen >

           word_trunc_type : type of truncation to be done on a sentence when number of words exceeds < word_maxlen >

           char_pad_value : padding value when padding a word

           word_pad_value : padding value when padding a sentence

        """

        self.alphabets = list('abcdefghijklmnopqrstuvwxyz')
        self.lookup = tf.keras.layers.StringLookup(max_tokens=27, vocabulary=self.alphabets)
        self.char_maxlen = char_maxlen
        self.char_padding_type = char_padding_type
        self.char_pad_value = char_pad_value
        self.char_trunc_type = char_trunc_type
        self.word_maxlen = word_maxlen
        self.word_padding_type = word_padding_type
        self.word_trunc_type = word_trunc_type
        self.word_pad_value = word_pad_value
        
    def to_char(self, sentence):
        sentence = [list(word) for word in sentence]
        return sentence

    def pad_words(self, inputs):
        return tf.keras.utils.pad_sequences(inputs, dtype=object, truncating=self.word_trunc_type,
                                            maxlen=self.word_maxlen, padding=self.word_padding_type,
                                            value=self.word_pad_value)

    def pad_chars(self, sentence):
        return tf.keras.utils.pad_sequences(sentence, truncating=self.char_trunc_type,
                                            maxlen=self.char_maxlen, padding=self.char_padding_type,
                                            value=self.char_pad_value)

    def tokenization(self, sentence):
        return sentence.split(" ")

    def __call__(self, inputs):
        inputs = [self.tokenization(input) for input in inputs]
        inputs = self.pad_words(inputs)
        inputs = [self.to_char(input) for input in inputs]
        inputs = self.lookup(tf.ragged.constant(inputs))
        inputs = inputs.to_list()
        inputs = [self.pad_chars(i) for i in inputs]
        return tf.Variable(inputs)




class BertED():
    def __init__(self, vocab, max_len):
        super(BertED, self).__init__()
        self.vocab = vocab
        self.maxlen = max_len
        self.encode = tf.keras.layers.TextVectorization(max_tokens=1000,
                                                        output_mode='int',
                                                        vocabulary=self.vocab,
                                                        standardize='lower_and_strip_punctuation'
                                                        )
        self.decode = tf.keras.layers.StringLookup(max_tokens=1000,
                                                   output_mode='int',
                                                   vocabulary=self.vocab,
                                                   invert=True)

    def pad(self, inputs):
        inputs = tf.keras.utils.pad_sequences(
            inputs,
            maxlen=self.maxlen,
            dtype='int32',
            padding='post',
            truncating='post',
            value=0
        )
        return tf.convert_to_tensor(inputs)

    def mask(self, input_tensor):
        mask_tensor = tf.where(tf.equal(input_tensor, 0), tf.fill(tf.shape(input_tensor), 0),
                               tf.ones_like(input_tensor))
        return mask_tensor

    def typeids(self, input):
        return tf.zeros_like(input, dtype=tf.int32)

    def create_dict(self, input):
        sample = dict()
        sample['input_word_ids'] = tf.cast(self.pad(self.encode(input)), tf.int32)
        sample['input_mask'] = self.mask(sample['input_word_ids'])
        sample['input_type_ids'] = self.typeids(sample['input_word_ids'])
        return sample

    def __call__(self, inputs):
        inputs = [self.create_dict(input) for input in inputs]
        return inputs

    def decoder(self, input):
        output = self.decode(input)
        cond = tf.math.logical_not(tf.equal(output, "[UNK]"))
        output = tf.boolean_mask(output, cond)
        return output

    def return_vocab(self):
        return self.encode.get_vocabulary()

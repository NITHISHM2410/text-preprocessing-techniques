import tensorflow as tf

class BertED(tf.keras.layers.Layer):
    def __init__(self, vocab, max_tokens=1_000_000, max_len=150):

        """
        vocab(String) : path of text file.

        max_len(Integer) : Maximum length of Input Sequence. Should be <= BERT Embedding dim(Default : 150)

        max_tokens : Maximum number of tokens to be considered from the Input vocabulary (Default 1,000,000)

        """
        super(BertED, self).__init__()

        self.maxlen = max_len
        self.max_tokens = max_tokens

        self._encode = tf.keras.layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_mode='int',
            vocabulary=self._get_vocab(vocab),
            output_sequence_length=self.maxlen,
            standardize='lower_and_strip_punctuation'
        )

        self._vocabulary = self.return_vocab()

    def _get_vocab(self, vocab):
        with open(vocab) as f:
            vocab = f.read()
        vocab = vocab.split("\n")
        vocab.insert(0,'[START]')
        vocab.insert(0,'[END]')
        vocab.insert(0,'[MASK]')
        return vocab

    def _mask(self, input_tensor):
        mask_tensor = tf.where(
            tf.equal(input_tensor, 0),
            tf.fill(tf.shape(input_tensor), 0),
            tf.ones_like(input_tensor))
        return mask_tensor

    def _typeids(self, input):
        return tf.zeros_like(input, dtype=tf.int32)

    def _create_dict(self, input):
        sample = dict()

        sample['input_word_ids'] = tf.cast(
            self._encode(input),
            tf.int32)

        sample['input_mask'] = self._mask(sample['input_word_ids'])

        sample['input_type_ids'] = self._typeids(sample['input_word_ids'])

        return sample

    def call(self, inputs):
        return self._create_dict(inputs)

    def _decoder(self, input):
        output = tf.gather(self._vocabulary, input)
        return output

    def back_to_string(self, inputs):
        decoded_sentences = tf.map_fn(self._decoder, inputs, dtype=tf.string)
        decoded_sentences = decoded_sentences.numpy().tolist()
        decoded_sentences = [((b" ".join(sentence)).decode('utf-8')).strip() for sentence in decoded_sentences]
        return decoded_sentences

    def return_vocab(self):
        return self._encode.get_vocabulary()


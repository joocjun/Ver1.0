# https://www.tensorflow.org/tutorials/text/transformer

import tensorflow as tf

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, hidden_size, head, masked=False):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.head = head  # head의 수

        self.wq = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.wk = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.wv = tf.keras.layers.Dense(hidden_size, use_bias=False)

        self.linear = tf.keras.layers.Dense(hidden_size, use_bias=False)

        self.scale = tf.keras.layers.Lambda(lambda x: x / np.sqrt(hidden_size))
        self.masked = masked

    def call(self, q, k, v, mask=None):

        assert q.shape[-1] % self.head == 0

        wq = tf.reshape(self.wq(q), [self.head, q.shape[0], q.shape[1], -1])
        wk = tf.reshape(self.wk(k), [self.head, k.shape[0], k.shape[1], -1])
        wv = tf.reshape(self.wv(v), [self.head, v.shape[0], v.shape[1], -1])
        # (head_n,bs,ts,hs/head_n)
        scaled_attention_logit = self.scale(tf.matmul(wq, wk, transpose_b=True))

        if self.masked:
            mask = (1 - tf.linalg.band_part(tf.ones(scaled_attention_logit.shape[1:]), -1, 0)) * -1e9
            scaled_attention_logit = tf.reshape([head + mask for head in scaled_attention_logit],
                                                 scaled_attention_logit.shape)

        attention_weight = tf.nn.softmax(scaled_attention_logit, axis=-1)
        # head_n,bs,ts,hs/head_n
        output = tf.reshape(tf.matmul(attention_weight, wv), q.shape)
        output = self.linear(output)
        # (bs,ts,hs)

        return attention_weight, output


class EncoderBlock(tf.keras.Model):
    def __init__(self, hidden_size, head_num, dropout, layer_norm_epsilon):
        super(EncoderBlock, self).__init__()

        self.MultiHeadAttention = MultiHeadAttention(hidden_size, head_num)
        self.MHA_Dropout = tf.keras.layers.Dropout(dropout)
        self.MHA_Normalization = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

        self.FFN = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size * 4),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_size)])

        self.FFN_Dropout = tf.keras.layers.Dropout(dropout)
        self.FFN_Normalization = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

    def call(self, x):
        normalized_x = self.MHA_Normalization(x)
        attention_weight, attention_output = self.MultiHeadAttention(normalized_x, normalized_x, normalized_x)
        attention_output = x + self.MHA_Dropout(attention_output)

        normalized_attention_output = self.FFN_Normalization(attention_output)
        FFN_output = attention_output + self.FFN_Dropout(self.FFN(normalized_attention_output))

        return attention_weight, FFN_output
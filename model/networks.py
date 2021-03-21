from abc import ABC

import tensorflow as tf
from models import EncoderBlock

class BERT(tf.keras.Model):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.Encoder = EncoderBlock(self.config.)
    def call(self, **kwargs):
        pass
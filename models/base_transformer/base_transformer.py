import os
from collections import OrderedDict

from tensor2tensor.bin import t2t_decoder
import tensorflow as tf

from models.ILanguageModel import ILanguageModel


class BaseTransformerModel(ILanguageModel):
    """Base transformer model.

    Usage sample:

    transformer = BaseTransformerModel(initial_context=['Hello', 'world', '.'])

    transformer.add_word_to_context('Test')

    next_word_ranking = transformer()
    """

    def __init__(
        self,
        context_window_length=16,
        next_word_possibilities_number=16,
        initial_context="",
    ):
        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number

        if initial_context:
            self.context = ''.join(initial_context)
        else:
            self.context = ''
		
        FLAGS = tf.flags.FLAGS
        FLAGS.problem = "languagemodel_lm1b32k"
        FLAGS.model = "transformer"
        FLAGS.hparams_set = "transformer_base"
        FLAGS.output_dir = "models/base_transformer"
        FLAGS.decode_to_file = "./decoded.txt"
        FLAGS.decode_hparams = "beam_size={},return_beams=True,extra_length=1".format(next_word_possibilities_number)
        FLAGS.data_dir = "./models/base_transformer"
        FLAGS.decode_from_file = "./input.txt"

    def reset(self, new_context):
        if len(new_context) > self.window_length:
            raise Exception('New context exceeds context window length.')

        self.context = new_context

    def add_word_to_context(self, word):
        assert len(self.context) <= self.window_length

        if len(self.context) == self.window_length:
            self.context.pop()

        self.context.append(word)

    def __str__(self):
        return "base_transformer"

    def __call__(self):
        if os.path.exists('input.txt'):
            os.remove('input.txt')
        if os.path.exists('decoded.txt'):
            os.remove('decoded.txt')
        with open('input.txt', 'w') as f:
            f.write(self.context)
        t2t_decoder.main('None')
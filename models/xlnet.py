from collections import OrderedDict

import torch
import transformers as tfms

from .ILanguageModel import ILanguageModel


class XLNetModel(ILanguageModel):
    """XLNet Language Model.

    Usage sample:

    xlnet = XLNetModel(initial_context=['Hello', 'world', '.'])

    xlnet.add_word_to_context('Test')

    next_word_ranking = xlnet()
    """

    def __init__(
        self,
        context_window_length=16,
        next_word_possibilities_number=16,
        initial_context="",
    ):
        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number
        self.model = tfms.XLNetLMHeadModel.from_pretrained("xlnet-large-cased")
        self.tokenizer = tfms.XLNetTokenizer.from_pretrained("xlnet-large-cased")

        # Prevent dropout from being considered when evaluating
        self.model.eval()

        if initial_context:
            self.context = initial_context
        else:
            self.context = []

    def reset(self, new_context):
        if len(new_context) > self.window_length:
            raise Exception("New context exceeds context window length.")

        self.context = new_context

    def add_word_to_context(self, word):
        assert len(self.context) <= self.window_length

        if len(self.context) == self.window_length:
            self.context.pop()

        self.context.append(word)

    def __call__(self):
        if len(self.context) > 0:
            inpt = self.tokenizer.encode(" ".join(self.context))
        else:
            inpt = self.tokenizer.encode("")

        with torch.no_grad():
            inpt = torch.tensor([inpt])
            outputs = self.model(inpt)
            loss = outputs[0][0, -1, :]
            softmaxed = torch.softmax(loss, dim=0)
            top_words = torch.topk(softmaxed, k=self.num_possibilities)
            top_words_indices = top_words.indices
            top_words_probabilities = top_words.values

            result = OrderedDict()
            for i in range(top_words_indices.shape[0]):
                word = self.tokenizer.decode(torch.tensor([top_words_indices[i]]))
                prob = top_words_probabilities[i].item()
                result[word] = prob

            return result


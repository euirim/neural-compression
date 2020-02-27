import torch
import transformers as tfms

from .ILanguageModel import ILanguageModel


class BERTModel(ILanguageModel):
    """BERT Language Model.

    Usage sample:

    bert = BERTModel(initial_context=['Hello', 'world', '.'])

    bert.add_word_to_context('Test')

    next_word_ranking = bert()
    """
    def __init__(self, context_window_length=16, next_word_possibilities_number=16, initial_context=''):
        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number
        self.model = tfms.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = tfms.BertForMaskedLM.from_pretrained('bert-base-uncased')
        
        if initial_context:
            self.context = initial_context
        else:
            self.context = []

    def reset(self, new_context):
        if len(new_context) > self.window_length:
            raise Exception('New context exceeds context window length.')

        self.context = new_context

    def add_word_to_context(self, word):
        assert len(self.context) <= self.window_length

        if len(self.context) == self.window_length:
            self.context.pop()

        self.context.append(word)

    def __call__(self):
        if len(self.context) > 0:
            inpt = self.tokenizer.encode(' '.join(self.context), add_prefix_space=True)
        else:
            inpt = self.tokenizer.encode('', add_prefix_space=True)

        inpt = torch.tensor(inpt)
        outputs = self.model(inpt, labels=inpt)
        loss, logits = outputs[:2]

        return self.tokenizer.decode(torch.tensor([torch.argmax(logits)]))

import torch
import transformers as tfms

from .ILanguageModel import ILanguageModel


class GPT2Model(ILanguageModel):
    """GPT-2 Language Model.

    Usage sample:

    gpt2 = GPT2Model(initial_context=['Hello', 'world', '.'])

    gpt2.add_word_to_context('Test')

    next_word_ranking = gpt2()
    """
    def __init__(self, context_window_length=16, next_word_possibilities_number=16, initial_context=''):
        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number
        self.model = tfms.GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = tfms.GPT2Tokenizer.from_pretrained('gpt2')

        # Prevent dropout from being considered when evaluating
        self.model.eval()
        
        if initial_context:
            self.context = initial_context
        else:
            self.context = []

    def reset(self, new_context):
        if len(new_context.split()) > self.window_length:
            raise Exception('New context exceeds context window length.')

        self.context = new_context.split()

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

        with torch.no_grad():
            inpt = torch.tensor([inpt])
            outputs = self.model(inpt)
            loss = outputs[0][0,-1,:]
            softmaxed = torch.softmax(loss, dim=0)
            top_words = torch.topk(softmaxed, k=self.num_possibilities)
            top_words_indices = top_words.indices
            top_words_probabilities = top_words.values

            result = []
            for i in range(top_words_indices.shape[0]):
                result.append((
                    self.tokenizer.decode(torch.tensor([top_words_indices[i]])),
                    top_words_probabilities[i].item()
                ))

            return result
 
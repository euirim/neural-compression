from collections import OrderedDict

import torch
import transformers as tfms

from .ILanguageModel import ILanguageModel


device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GPTModel(ILanguageModel):
    """GPT Language Model.

    Usage sample:

    gpt = GPTModel(initial_context=['Hello', 'world', '.'])

    gpt.add_word_to_context('Test')

    next_word_ranking = gpt()
    """

    def __init__(
        self,
        context_window_length=16,
        next_word_possibilities_number=16,
        initial_context=None,
    ):
        self.name = "GPT"
        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number
        self.model = tfms.OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(device)
        self.tokenizer = tfms.OpenAIGPTTokenizer.from_pretrained("openai-gpt")

        # Prevent dropout from being considered when evaluating
        self.model.eval()

        if initial_context is not None:
            self.context = initial_context
        else:
            self.context = []

    def reset(self, new_context):
        if len(new_context) > self.window_length:
            print(
                f"New context ({len(new_context)}) exceeds context window length ({self.window_length})."
            )
            new_context = new_context[-self.window_length:]

        self.context = new_context

    def add_word_to_context(self, word):
        assert len(self.context) <= self.window_length

        if len(self.context) == self.window_length:
            self.context.pop()

        self.context.append(word)

    def __str__(self):
        return "GPT"

    def __call__(self):
        if len(self.context) > 0:
            inpt = self.tokenizer.encode(" ".join(self.context))
        else:
            inpt = self.tokenizer.encode("")

        with torch.no_grad():
            inpt = torch.tensor([inpt]).to(device)
            outputs = self.model(inpt)
            loss = outputs[0][0, -1, :]
            softmaxed = torch.softmax(loss, dim=0)
            top_words = torch.topk(softmaxed, k=self.num_possibilities)
            top_words_indices = top_words.indices
            top_words_probabilities = top_words.values

            result = OrderedDict()
            for i in range(top_words_indices.shape[0]):
                word = self.tokenizer.decode(
                    torch.tensor([top_words_indices[i]]).to(device)
                )
                prob = top_words_probabilities[i].item()
                result[word] = prob

            return result


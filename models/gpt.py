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

    gpt = GPTModel(
        initial_context=[
            [12, 16, 17],
            [13, 14, 15],
            [128, 767, 11],
        ]
    )

    gpt.add_to_context([12, 14, 11])

    next_word_ranking = gpt()
    """

    def __init__(
        self,
        context_window_length=16,
        next_word_possibilities_number=16,
        initial_context=None,
        num_batches=4,
    ):
        self.num_batches = num_batches
        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number
        self.model = tfms.OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(device)
        self.tokenizer = tfms.OpenAIGPTTokenizer.from_pretrained("openai-gpt")

        # Prevent dropout from being considered when evaluating
        self.model.eval()

        if initial_context is not None:
            assert len(initial_context) == num_batches
            self.context = initial_context
        else:
            self.context = [self.tokenizer.encode("") for _ in range(num_batches)]

    def reset(self, new_context):
        assert len(new_context) == self.num_batches
        if len(new_context) > self.window_length:
            raise Exception("New context exceeds context window length.")

        self.context = new_context

    def add_to_context(self, words):
        assert len(words) == len(self.context)

        for r in range(len(self.context)):
            assert len(self.context[r]) <= self.window_length

            if len(self.context[r]) == self.window_length:
                self.context[r].pop()

            self.context[r].append(words[r])

    def __str__(self):
        return "GPT"

    def __call__(self):
        inpt = self.context

        self.model.eval()
        with torch.no_grad():
            print(inpt)
            inpt = torch.tensor([inpt]).to(device)
            print("inpt size", inpt.size())
            outputs = self.model(inpt)
            print("outputs", outputs[0])
            print("outputs size", outputs[0].size())
            loss = outputs[0][0].narrow(1, -1, 1).squeeze(dim=1)
            print("loss size", loss.size())
            softmaxed = torch.softmax(loss, dim=-1)
            print("softmaxed size", softmaxed.size())
            top_words = torch.topk(softmaxed, k=self.num_possibilities)
            print("top_words indices size", top_words.indices.size())
            top_words_indices = top_words.indices
            top_words_probabilities = top_words.values

            result = OrderedDict()
            for c in range(top_words_indices.size(0)):
                for i in range(top_words_indices.size(1)):
                    word = top_words_indices[c][i].item()
                    prob = top_words_probabilities[c][i].item()

                    result[(c, word)] = prob

            return result

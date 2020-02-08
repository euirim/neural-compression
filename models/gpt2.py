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
        if initial_context:
            self.context = initial_context
        else:
            self.context = []

        self.window_length = context_window_length
        self.num_possibilities = next_word_possibilities_number

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
        model = tfms.RobertaModel.from_pretrained('roberta-base')
        tokenizer = tfms.RobertaTokenizer.from_pretrained('roberta-base')
        #tokenizer.mask_token = '<mask>'
        filler = tfms.FillMaskPipeline(
            model=model,
            tokenizer=tokenizer,
            topk=self.num_possibilities,
        )
        return filler(' '.join(self.context + [tokenizer.mask_token]))
        
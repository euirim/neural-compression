from models.gpt2 import GPT2Model
from models.xlnet import XLNetModel
from models.gpt import GPTModel


def main():
    gpt2 = GPT2Model(initial_context=["Hello", "the"])
    gpt2.add_word_to_context("world")
    next_word_ranking = gpt2()
    print(next_word_ranking)

    xlnet = XLNetModel(initial_context=["Hello", "the"])
    xlnet.add_word_to_context("world")
    next_word_ranking = xlnet()
    print(next_word_ranking)

    gpt = GPTModel(initial_context=["Hello", "the"])
    gpt.add_word_to_context("world")
    next_word_ranking = gpt()
    print(next_word_ranking)


if __name__ == "__main__":
    main()

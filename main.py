from models.gpt2 import GPT2Model
from models.xlnet import XLNetModel
from models.gpt import GPTModel


def main():
    """
    gpt2 = GPT2Model(initial_context=["Hello", "the"])
    gpt2.add_word_to_context("world")
    next_word_ranking = gpt2()
    print(next_word_ranking)

    xlnet = XLNetModel(initial_context=["Hello", "the"])
    xlnet.add_word_to_context("world")
    next_word_ranking = xlnet()
    print(next_word_ranking)
    """

    gpt = GPTModel(initial_context=None, num_batches=2)
    gpt.reset(
        [gpt.tokenizer.encode("The New York"), gpt.tokenizer.encode("The Goldman"),]
    )
    # gpt.add_to_context([15, 20, 13])
    next_word_ranking = gpt()
    print(next_word_ranking)
    for k in next_word_ranking:
        c, w_i = k
        print(gpt.tokenizer.decode([w_i]))


if __name__ == "__main__":
    main()

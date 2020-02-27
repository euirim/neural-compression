from models.gpt2 import GPT2Model

def main():
    gpt2 = GPT2Model(initial_context=['Hello', 'the'])
    gpt2.add_word_to_context('world')
    next_word_ranking = gpt2()
    print(next_word_ranking)

if __name__=='__main__':
    main()
from abc import ABC, abstractmethod


class ILanguageModel(ABC):
    """Interface for a generic language model
    
    Usage sample:

    lm = LanguageModelImplementation(initial_context=['Hello', 'world', '.'])

    lm.add_word_to_context('Test')

    next_word_ranking = lm()
    """

    @abstractmethod
    def __init__(
        self,
        initial_context,
        context_window_length=16,
        next_word_possibilities_number=16,
    ):
        pass

    def get_context_window_length(self):
        return self._context_window_length

    @abstractmethod
    def reset(self, new_context):
        """
        Resets the context.
        @param new_context: String.
        """
        pass

    @abstractmethod
    def add_word_to_context(self, word):
        """
        Adds the given word to the context, and removes oldest word if necessary.
        @param word: String.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Returns the name of the model as a string.
        """
        pass

    @abstractmethod
    def __call__(self):
        """
        Gets a ranking of possible next words, given current context.
        @returns Ordered dictionary with the word as a key and probability as a value.
        """
        pass

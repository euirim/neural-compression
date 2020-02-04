from abc import ABC, abstractmethod

class ILanguageModel(ABC):
	"""Interface for a generic language model"""


	@abstractmethod
	def __init__(self, context_window_length=16, next_word_possibilities_number=16):
		pass

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
	def __call__(self):
		"""
		Gets a ranking of possible next words, given current context.
		@returns Dict<word, probability>.
		"""
		pass

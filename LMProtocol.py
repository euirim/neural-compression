from .ILanguageModel import ILanguageModel
from collections import OrderedDict

class LMProtocol:

	def __init__(self, language_model, context_window_length=16, next_word_possibilities_number=16):
		"""
		@param language_model: ILanguageModel
		"""
		self.lm = language_model(context_window_length=16, next_word_possibilities_number=16)
		self._context_window_length = context_window_length
		self._next_word_possibilities_number = next_word_possibilities_number


	def compress(self, text):
		"""
		@param text: String to compress
		@returns binary string
		"""

		compressed_object = self._get_compressed_object(text)

	def _get_compressed_object(self, text):
		"""
		@param text: String to compress
		@returns Dict:
			initial_context: String
			words: List<Dict:
				out_of_vocabulary: Bool
				word: String (only if out_of_vocabulary)
				ranking: Int (only if not out_of_vocabulary)
			>
		"""
		words = text.split()
		assert len(words) > self._context_window_length
		compressed_object = {}
		compressed_object['initial_context'] = ' '.join(words[:self._context_window_length])

		self.lm.reset(compressed_object['initial_context'])

		compressed_object['words'] = []
		for word in words[self._context_window_length :]:
			word_probabilities = self.lm()
			if word not in word_probabilities:
				compressed_object['words'].append({'out_of_vocabulary': True, 'word': word})
			else:
				ranking = self._get_ranking_from_probabilities(word_probabilities, word)
				compressed_object['words'].append({'out_of_vocabulary': False, 'ranking': ranking})
			self.lm.add_word_to_context(word)
		return compressed_object

	def _get_string_from_compressed_object(self, compressed_object):
		"""
		@param compressed_object: Dict:
			initial_context: String
			words: List<Dict:
				out_of_vocabulary: Bool
				word: String (only if out_of_vocabulary)
				ranking: Int (only if not out_of_vocabulary)
			>
		@returns uncompressed_string
		"""
		self.lm.reset(compressed_object['initial_context'])
		words = []

		for item in compressed_object['words']:
			if item['out_of_vocabulary']:
				word = item['word']
			else:
				word_probabilities = self.lm()
				word = list(word_probabilities.keys())[item['ranking']]
			words.append(word)
			self.lm.add_word_to_context(word)
		return ' '.join(compressed_object['initial_context'] + words)


	def _get_ranking_from_probabilities(_, word_probabilities, word):
		"""
		@param word_probabilities: Dict<word, probability>.
		@returns ranking: Int
		"""
		ordered_words = OrderedDict(sorted(word_probabilities.items(), key=lambda t: -t[1]))
		return list(ordered_words.keys()).index(word)

	def _get_binary_from_object(self, compressed_object):


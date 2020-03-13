from collections import OrderedDict
import math
from bitarray import bitarray

import zlib
from tqdm import tqdm

from models import ILanguageModel


class LMProtocol:
    def __init__(
        self,
        language_model,
        context_window_length=16,
        next_word_possibilities_number=16,
        out_of_vocabulary_word_max_bit_size=1024,
        initial_context_max_bit_size=16384,
    ):
        """
        @param language_model: ILanguageModel
        """
        assert math.log2(next_word_possibilities_number).is_integer()
        assert math.log2(out_of_vocabulary_word_max_bit_size).is_integer()
        assert math.log2(initial_context_max_bit_size).is_integer()
        self.lm = language_model(
            context_window_length=context_window_length,
            next_word_possibilities_number=next_word_possibilities_number,
        )
        self._context_window_length = context_window_length
        self._next_word_possibilities_number = next_word_possibilities_number
        self._out_of_vocabulary_word_max_bit_size = out_of_vocabulary_word_max_bit_size
        self._initial_context_max_bit_size = initial_context_max_bit_size

    def compress(self, text):
        """
        @param text: String to compress
        @returns binary string
        """

        compressed_object = self._get_compressed_object(text)
        binary = self._get_binary_from_object(compressed_object)
        zlib_binary = zlib.compress(binary.tobytes())

        return zlib_binary

    def decompress(self, compressed_binary):
        """
        @param compressed_binary: binary string
        @returns original string
        """
        binary = bitarray()
        binary.frombytes(zlib.decompress(compressed_binary))
        compressed_object = self._get_object_from_binary(binary)
        uncompressed_string = self._get_string_from_compressed_object(compressed_object)
        return uncompressed_string

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
        compressed_object["initial_context"] = " ".join(
            words[: self._context_window_length]
        )

        self.lm.reset(compressed_object["initial_context"].split())

        compressed_object["words"] = []
        print("Getting compressed object.")
        for word in tqdm(words[self._context_window_length :]):
            word_probabilities = self.lm()
            if word not in word_probabilities:
                compressed_object["words"].append(
                    {"out_of_vocabulary": True, "word": word}
                )
            else:
                ranking = self._get_ranking_from_probabilities(word_probabilities, word)
                compressed_object["words"].append(
                    {"out_of_vocabulary": False, "ranking": ranking}
                )
            self.lm.add_word_to_context(word)
        return compressed_object

    def _get_binary_from_object(self, compressed_object):
        binary = bitarray()
        binary_initial_context = bitarray()
        initial_context_bytes = compressed_object["initial_context"].encode("utf-8")
        initial_context_bits = bitarray()
        initial_context_bits.frombytes(initial_context_bytes)
        binary_initial_context.extend(
            (
                "{0:0" + str(int(math.log2(self._initial_context_max_bit_size))) + "b}"
            ).format(len(initial_context_bits))
        )
        binary_initial_context.frombytes(initial_context_bytes)
        binary.extend(binary_initial_context)

        print("Getting binary from object.")
        for word in tqdm(compressed_object["words"]):
            binary_word = bitarray()
            # TODO: instead of having a separate flag for out of vocabulary, use ranking 0 to mean o.o.v.
            if word["out_of_vocabulary"]:
                binary_word.append(True)
                uncompressed_word = bitarray()
                uncompressed_word.frombytes(word["word"].encode("utf-8"))
                binary_word.extend(
                    (
                        "{0:0"
                        + str(int(math.log2(self._out_of_vocabulary_word_max_bit_size)))
                        + "b}"
                    ).format(len(uncompressed_word))
                )
                binary_word.extend(uncompressed_word)
            else:
                binary_word.append(False)
                binary_word.extend(
                    (
                        "{0:0"
                        + str(int(math.log2(self._next_word_possibilities_number)))
                        + "b}"
                    ).format(word["ranking"])
                )

            binary.extend(binary_word)

        # Append 1's to the end
        binary.extend([True for _ in range(8 - len(binary) % 8)])
        return binary

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
        self.lm.reset(compressed_object["initial_context"].split())
        words = []

        print("Getting string from compressed object.")
        for item in tqdm(compressed_object["words"]):
            if item["out_of_vocabulary"]:
                word = item["word"]
            else:
                word_probabilities = self.lm()
                word = list(word_probabilities.keys())[item["ranking"]]
            words.append(word)
            self.lm.add_word_to_context(word)
        return compressed_object["initial_context"] + " ".join(words)

    def _get_ranking_from_probabilities(self, word_probabilities, word):
        """
        @param word_probabilities: Dict<word, probability>
        @returns ranking: Int
        """
        ordered_words = OrderedDict(
            sorted(word_probabilities.items(), key=lambda t: -t[1])
        )
        return list(ordered_words.keys()).index(word)

    def _get_object_from_binary(self, binary):
        obj = {}
        initial_context_length_binary = binary[
            0 : int(math.log2(self._initial_context_max_bit_size))
        ]
        initial_context_length = int(initial_context_length_binary.to01(), 2)
        initial_context_binary = binary[
            int(math.log2(self._initial_context_max_bit_size)) : int(
                math.log2(self._initial_context_max_bit_size)
            )
            + initial_context_length
        ]
        obj["initial_context"] = initial_context_binary.tobytes().decode("utf-8")
        obj["words"] = []
        words_binary = binary[
            int(math.log2(self._initial_context_max_bit_size))
            + initial_context_length :
        ]
        while len(words_binary) > 0:
            word = {}
            word["out_of_vocabulary"] = words_binary[0]
            if word["out_of_vocabulary"]:
                # If there are fewer than 8 bits, then it must be the end of the file, ignore bits.
                if len(words_binary) < 8:
                    words_binary = []
                    word = None
                else:
                    word_binary_length_binary = words_binary[
                        1 : 1
                        + int(math.log2(self._out_of_vocabulary_word_max_bit_size))
                    ]
                    word_binary_length = int(word_binary_length_binary.to01(), 2)
                    word_binary = words_binary[
                        1
                        + int(math.log2(self._out_of_vocabulary_word_max_bit_size)) : 1
                        + int(math.log2(self._out_of_vocabulary_word_max_bit_size))
                        + word_binary_length
                    ]
                    word["word"] = word_binary.tobytes().decode("utf-8")
                    words_binary = words_binary[
                        1
                        + int(math.log2(self._out_of_vocabulary_word_max_bit_size))
                        + word_binary_length :
                    ]
            else:
                ranking_binary = words_binary[
                    1 : 1 + int(math.log2(self._next_word_possibilities_number))
                ]
                word["ranking"] = int(ranking_binary.to01(), 2)
                words_binary = words_binary[
                    1 + int(math.log2(self._next_word_possibilities_number)) :
                ]
            if word:
                obj["words"].append(word)

        return obj


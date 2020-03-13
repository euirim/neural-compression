import json
import time

import torch

from corpus import Corpus
from models.gpt import GPTModel
from models.gpt2 import GPT2Model
from models.xlnet import XLNetModel
from LMProtocol import LMProtocol

# from models.base_transformer.base_transformer import BaseTransformerModel


device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_corpus(name, filename, preprocess_func):
    corpus = Corpus(name=name, filename=filename, preprocess_func=preprocess_func)
    corpus.load()
    return corpus


def eval_model(model, corpus, model_name):
    """
    @returns Dict:
        model_name: Dict:
            (window_length, num_next_word):
                duration: Int (seconds)
                compression_rate: Float (between 0 and 1)
    """
    context_window_lengths = [2, 8, 32, 128]
    next_word_possibilities_numbers = [2, 8, 32, 128]
    total_experiments = len(context_window_lengths) * len(
        next_word_possibilities_numbers
    )
    result = {}

    # Run experiment on given model and store result
    # in given results object.
    with open(f"results/results_{model_name.lower()}.txt", "w+") as f:
        for i, cwl in enumerate(context_window_lengths, 1):
            for j, nwpn in enumerate(next_word_possibilities_numbers, 1):
                exp_number = (i - 1) * len(next_word_possibilities_numbers) + j
                print(f"Experiment ({exp_number} / {total_experiments})")
                protocol = LMProtocol(
                    language_model=model,
                    context_window_length=cwl,
                    next_word_possibilities_number=nwpn,
                )
                start_time = time.time()
                compressed_str = protocol.compress(corpus.docs)
                end_time = time.time()
                time_elapsed = end_time - start_time
                compression_rate = len(compressed_str) / len(corpus.docs)

                f.write(
                    f"CWL: {cwl} | NWPN: {nwpn} | duration: {time_elapsed} | compression_rate: {compression_rate}\n"
                )

                result[f"{cwl} | {nwpn}"] = {
                    "duration": time_elapsed,
                    "compression_rate": compression_rate,
                }

    return result


def output_results(results):
    """
    Pretty print summary of given results.
    """
    with open("results/results_final.json", "w+") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    models = [
        ("GPT", GPTModel),
        ("GPT-2", GPT2Model),
        ("XLNet", XLNetModel),
    ]
    corpus_filename = "data/full.txt"

    # Load corpus
    print("Loading corpus.")
    corpus = load_corpus("Full", corpus_filename, lambda x: x)

    # Run experiments on each model
    print("Evaluating models.")
    results = {}

    for model_name, model in models:
        print("-" * 80)
        print(f"Evaluating {model_name}.")
        print("-" * 80)
        result = eval_model(model, corpus, model_name)
        results[model_name] = result

    # Report results
    print(f"Reporting results.")
    output_results(results)

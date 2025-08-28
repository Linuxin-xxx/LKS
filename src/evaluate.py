import argparse
import os
import re
import torch
import datasets
import numpy as np
import dotenv
import nltk
import unicodedata
from pathlib import Path
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer
from log_util import create_logger
import model_handler
import data_handler

# nltk.download('punkt_tab')

MODEL_CHOICES = ["llama2", "mistral"]
RESULTS_DIR = "../results/"
EVAL_TYPES = ["reliability", "generality", "locality"]
DEFAULT_MAX_NEW_TOKENS = 100
KE_ZSRE_DIR = "../data/zsre/"
KE_CF_DIR = "../data/counterfact/"

logger = create_logger()

dotenv.load_dotenv()
EVALUATION_DATASET = os.getenv('EVALUATION_DATASET')
logger.info(f"EVALUATION_DATASET: {EVALUATION_DATASET}")
if EVALUATION_DATASET == "counterfact":
    DEFAULT_EVAL_DATASET = KE_CF_DIR + "cf_eval_data.json"
elif EVALUATION_DATASET == "zsre":
    DEFAULT_EVAL_DATASET = KE_ZSRE_DIR + "zsre_eval_data.json"
else:
    DEFAULT_EVAL_DATASET = ""


def n_gram_entropy(gen_texts):
    gen_texts = [
        unicodedata.normalize("NFKD", text)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for text in gen_texts
    ]

    res = []
    for text in gen_texts:
        ns = [2, 3]
        weights = [2 / 3, 4 / 3]

        entropy_list = []
        for n in ns:
            tokens = nltk.word_tokenize(text)
            ngrams = nltk.ngrams(tokens, n)
            fdist = nltk.FreqDist(ngrams)

            freqs = np.array([freq for _, freq in fdist.items()])
            freqs = freqs / freqs.sum()

            entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

        entropy_list = np.array(entropy_list) * np.array(weights)

        res.append(np.mean(entropy_list))

    fluency = np.mean(res).item()

    return fluency


def get_score(model, tokenizer, batch, modifier, eval_args, eval_metric):
    target_ids_batch = batch.get("target_ids_batch", [])

    if eval_args.verbose:
        target_batch = batch.get("target_batch", [])
        print("target_batch: ", target_batch)

    if isinstance(batch.get("feature_hs_batch", 0), torch.Tensor):
        output_logits = model_handler.output_with_new_kb(model, modifier, batch)
    else:
        output_logits = model_handler.output_with_ori_model(model, batch.get("prompt_ids_batch"))

    top_id_batch = model_handler.get_top_token_ids(output_logits, batch)
    if eval_args.verbose:
        model_handler.get_output_tokens_batch(tokenizer, top_id_batch, desc="mdf")

    # locality
    if eval_metric == "locality":
        output_logits_ori = model_handler.output_with_ori_model(model, batch.get("prompt_ids_batch"))
        top_id_ori_batch = model_handler.get_top_token_ids(output_logits_ori, batch)
        if eval_args.verbose:
            model_handler.get_output_tokens_batch(tokenizer, top_id_ori_batch, desc="ori")

        score = 0
        for i in range(len(top_id_batch)):
            score += np.mean(np.equal(top_id_ori_batch[i], top_id_batch[i]))
        score /= len(top_id_batch)
        if eval_args.verbose:
            print("score: ", score)

        return score

    # reliability generality
    score = 0
    for i in range(len(top_id_batch)):
        score += np.mean(np.equal(target_ids_batch[i].cpu().numpy(), top_id_batch[i]))
    score /= len(top_id_batch)
    if eval_args.verbose:
        print("score: ", score)

    return score


def get_eval_dataset(eval_metric, eval_args):
    source_dataset = datasets.load_dataset("json", data_files=eval_args.evaluation_dataset,
                                           split=eval_args.data_split)

    dataset = []
    if eval_metric == "reliability":
        for data_dict in source_dataset:
            new_item = {"prompt": data_dict.get("prompt", ""),
                        "target": data_dict.get("target", "")
                        }
            dataset.append(new_item)
    elif eval_metric == "generality":
        for data_dict in source_dataset:
            new_item = {"prompt": data_dict.get("generality", ""),
                        "target": data_dict.get("target", "")
                        }
            dataset.append(new_item)
    elif eval_metric == "locality":
        for data_dict in source_dataset:
            new_item = {"prompt": data_dict.get("locality", {}).get("prompt", ""),
                        "target": data_dict.get("locality", {}).get("target", "")
                        }
            dataset.append(new_item)
    else:
        raise ValueError("Unknown eval_metric: ", eval_metric)

    return dataset


def evaluate_metrics(model, tokenizer, modifier, eval_args, eval_metric):
    logger.info(f"----- Evaluating {eval_metric} -----")

    dataset = get_eval_dataset(eval_metric, eval_args)

    # load edit_scope
    edit_scope = data_handler.load_json(Path(eval_args.load_path, "edit_scope.json"))

    eval_score = 0
    modifier.eval()
    with torch.inference_mode():
        for data_item in tqdm(dataset):
            data_item = data_handler.prepare_data_eval(model, tokenizer, data_item, modifier.hook_module, edit_scope)
            score = get_score(model, tokenizer, data_item, modifier, eval_args, eval_metric)
            eval_score += score

    final_score = eval_score / len(dataset)
    logger.info(f"\n--- Final {eval_metric} score: {final_score}\n")

    return final_score


def generation_mode(model, tokenizer, modifier, eval_args):
    logger.info(f"----- generation_mode -----")

    source_dataset = datasets.load_dataset("json", data_files=eval_args.evaluation_dataset, split=eval_args.data_split)
    dataset = []
    for data_dict in source_dataset:
        new_item = {"prompt": data_dict.get("prompt", ""),
                    "target": data_dict.get("target", "")
                    }
        dataset.append(new_item)

    # load edit_scope
    edit_scope = data_handler.load_json(Path(eval_args.load_path, "edit_scope.json"))

    fluency_ori = 0
    fluency = 0
    modifier.eval()
    with torch.inference_mode():
        for batch in tqdm(dataset):
            new_batch = data_handler.prepare_data_generation(model, tokenizer, batch, modifier.hook_module, edit_scope)
            if eval_args.verbose:
                target = new_batch.get("target", "")
                logger.info(f"target: {target}")
            _, output_ori = model_handler.generate_with_ori_model(model, tokenizer, new_batch.get("prompt_ids_batch"),
                                                                  eval_args.max_new_tokens, eval_args.verbose)

            if isinstance(batch.get("feature_hs_batch", 0), torch.Tensor):
                _, output = model_handler.generate_with_new_kb(model, tokenizer, modifier, new_batch,
                                                                     eval_args.max_new_tokens, eval_args.verbose)
            else:
                output = output_ori
                logger.info("mdf output: same as original model output")

            # fluency
            fluency_ori += n_gram_entropy(output_ori)
            fluency += n_gram_entropy(output)

    fluency_ori /= len(dataset)
    fluency /= len(dataset)
    logger.info(f"==> fluency_ori: {fluency_ori} fluency: {fluency}")

    return fluency_ori, fluency


def start_evaluate(eval_args):
    logger.info(f"*** Start Evaluation ***")

    if not eval_args.load_path:
        raise ValueError("Not input load_path")

    dir_name = Path(eval_args.load_path).name
    pattern = r'([0-9a-z\-]+)\_modifier\_([0-9a-zA-Z\_\-]+)\_([0-9]+)\-([0-9]+)_[0-9]{8}\_[0-9a-zA-Z\_\-]+'
    if re.search(pattern, dir_name):
        search_res = re.search(pattern, dir_name)
        eval_args.model = search_res.group(1)
        eval_args.module = search_res.group(2).replace("-", ".")
        # data_split
        start_i = int(search_res.group(3))
        end_i = int(search_res.group(4))
        input_search_res = re.search(r'\[([0-9]*):([0-9]*)]', eval_args.data_split)
        input_start_i = input_search_res.group(1)
        input_end_i = input_search_res.group(2)
        if not input_start_i or int(input_start_i) < start_i:
            input_start_i = start_i
        if not input_end_i or int(input_end_i) > end_i:
            input_end_i = end_i
        eval_args.data_split = f"train[{max(start_i, int(input_start_i))}:{min(end_i, int(input_end_i))}]"

    logger.info(f"model_name: {eval_args.model}")
    logger.info(f"module: {eval_args.module}")
    logger.info(f"data_split: {eval_args.data_split}\n")

    # load model & tokenizer
    device = eval_args.device or "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_handler.load_model(eval_args.model, device, eval_args.half_precision, eval_args.local)

    # create modifier
    modifier = model_handler.create_modifier(model, eval_args.module, eval_args.half_precision,
                                             Path(eval_args.load_path, "modifier.pth"))

    # eval
    if eval_args.generation:
        generation_mode(model, tokenizer, modifier, eval_args)
        return

    if "reliability" in eval_args.eval_type:
        evaluate_metrics(model, tokenizer, modifier, eval_args, "reliability")
    if "generality" in eval_args.eval_type:
        evaluate_metrics(model, tokenizer, modifier, eval_args, "generality")
    if "locality" in eval_args.eval_type:
        evaluate_metrics(model, tokenizer, modifier, eval_args, "locality")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation")

    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES, help="model to apply the modifier")
    parser.add_argument("--device", help="training device")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="set precision to torch.bfloat16")
    parser.add_argument("--local", type=bool, default=True, help="if load model from local dir")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbose")
    parser.add_argument("--load-path", type=str, default="", help="the path where to load the modifier")
    parser.add_argument("--module", type=str, default="model.layers.16", help="target module to modify")
    parser.add_argument("--eval-type", type=str, default="reliability", nargs='+', help="evaluation types")
    parser.add_argument("--evaluation-dataset", type=str, default=DEFAULT_EVAL_DATASET,
                        help="evaluation dataset will be derive from this path")
    parser.add_argument("--data-split", type=str, default="[:]", help="dataset split")
    parser.add_argument("-g", "--generation", action="store_true", default=False, help="evaluate in generation mode")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="when generate, max new tokens")

    args = parser.parse_args()

    start_evaluate(args)

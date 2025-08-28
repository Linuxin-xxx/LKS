import argparse
import os
import time
import torch
import datasets
import re
import dotenv
from pathlib import Path
import model_handler
from data_handler import generate_edit_scope, build_train_dataset
from log_util import create_logger

logger = create_logger()

# load env
dotenv.load_dotenv()
TRAIN_DATASET = os.getenv('TRAIN_DATASET')
logger.info("TRAIN_DATASET: %s", TRAIN_DATASET)

MODEL_CHOICES = ["llama2", "mistral"]
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCH = 20
DEFAULT_LR = 1e-4
DEFAULT_P_EDIT = 1
DEFAULT_P_EQ = 1
DEFAULT_P_LOC = 1
DEFAULT_P_STM = 2
RESULTS_DIR = Path("../results", TRAIN_DATASET)

if TRAIN_DATASET == "zsre":
    DEFAULT_TRAIN_DS_PATH = "../data/zsre/zsre_train_data.json"
elif TRAIN_DATASET == "counterfact":
    DEFAULT_TRAIN_DS_PATH = "../data/counterfact/cf_train_data.json"
else:
    DEFAULT_TRAIN_DS_PATH = ""


def start_train(train_args):
    logger.info("*** Start Training ***")

    # load model & tokenizer
    logger.info("Load_model: %s", train_args.model)
    device = train_args.device or "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_handler.load_model(train_args.model, device, train_args.half_precision, train_args.local)
    if "mistral" in train_args.model:
        train_args.batch_size = 1

    # load dataset
    datasets.disable_caching()
    dataset = datasets.load_dataset("json", data_files=train_args.train_data, split=train_args.data_split)
    train_args.data_num = len(dataset)
    datasets_dict = build_train_dataset(dataset, train_args.batch_size, train_args.loc_num)
    logger.info("data_num: %d", train_args.data_num)
    logger.info("batch_size: %d", train_args.batch_size)
    logger.info("p-edit: %d  p-eq: %d  p-loc: %d  loc-num: %d\n", train_args.p_edit, train_args.p_eq, train_args.p_loc,
                train_args.loc_num)

    # make save dir
    search_res = re.search(r'train\[([0-9]*):([0-9]*)]', train_args.data_split)
    start_i = search_res.group(1)
    end_i = search_res.group(2)
    if not start_i:
        start_i = 0
    if not end_i:
        end_i = int(start_i) + train_args.data_num
    split_num = f"{start_i}-{end_i}"
    module_ = train_args.module.replace(".", "-")
    time_str = str(int(time.time()))[:8]

    save_dir = Path(train_args.save_dir, f"{train_args.model}_modifier_{module_}_"
                                         f"{split_num}_{time_str}_{TRAIN_DATASET}")
    os.makedirs(save_dir, exist_ok=True)

    modifier_save_path = Path(save_dir, "modifier.pth")

    # save edit scope
    edit_scope = generate_edit_scope(dataset, save_dir)

    # create modifier
    modifier = model_handler.create_modifier(model, train_args.module, train_args.half_precision, train_args.load_path)

    # train and save modifier
    model_handler.train_modifier(modifier, model, tokenizer, datasets_dict, train_args, edit_scope, modifier_save_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modifier")

    parser.add_argument("--model", default="llama2", choices=MODEL_CHOICES, help="model to apply the modifier")
    parser.add_argument("--device", help="training device")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="set precision to torch.bfloat16")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbose")
    parser.add_argument("--local", type=bool, default=True, help="if load model from local dir")
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DS_PATH,
                        help="training dataset will be derive from this json file")
    parser.add_argument("--data-split", type=str, default="train[:100]", help="dataset split")
    parser.add_argument("--module", type=str, default="model.layers.16", help="target module to modify")
    parser.add_argument("--load-path", type=str, default="", help="the path where to load the modifier")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size")
    parser.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH, help="max train epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate")
    parser.add_argument("--save-dir", type=str, default=RESULTS_DIR, help="the dir where to save the modifier")
    parser.add_argument("--p-edit", type=float, default=DEFAULT_P_EDIT, help="edit parameter")
    parser.add_argument("--p-eq", type=float, default=DEFAULT_P_EQ, help="equivalence neighborhood parameter")
    parser.add_argument("--p-loc", type=float, default=DEFAULT_P_LOC, help="locality parameter")
    parser.add_argument("--loc-num", type=int, default=2, help="number of locality prompts in training")

    args = parser.parse_args()

    start_train(args)

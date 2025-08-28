import torch
import dotenv
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_scheduler
from nethook import Trace
from torch import nn, optim
from tqdm.auto import tqdm
import torch.nn.functional as F
import data_handler

logger = logging.getLogger("main")

# model path
LLAMA2_NAME = "meta-llama/Llama-2-7b-chat-hf"
MISTRAL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# TODO update to your own local dir
LLAMA2_NAME_LOCAL = "/your_local_dir/Llama-2-7b-chat-hf"
MISTRAL_NAME_LOCAL = "/your_local_dir/Mistral-7B-Instruct-v0.3"


# get access token
dotenv.load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


# create modifier
def create_modifier(model, module, half_precision, load_path=""):
    in_out_size = model.config.hidden_size

    if "gpt2" in str(type(model)):
        if "c_attn" in module:
            in_out_size = 4800
        elif "c_fc" in module:
            in_out_size = 6400
    else:
        for item in ["up_proj", "gate_proj", "act_fn"]:
            if item in module:
                in_out_size = model.config.intermediate_size
                break

    modifier = LinearM(in_out_size, module)
    # modifier = MLPM(in_out_size, module)
    modifier.to(model.device)
    logger.info(f"modifier: {modifier}")

    if load_path:
        load_modifier(modifier, load_path, model.device)

    # when model is set to torch.bfloat16
    if half_precision:
        logger.info("create modifier in torch.bfloat16")
        modifier.to(torch.bfloat16)

    return modifier


def save_modifier(best_state_dict, save_path):
    torch.save(best_state_dict, save_path)
    logger.info(f"+++ Save modifier.state_dict() at {save_path}")


def load_modifier(modifier, save_path, device):
    state_dict = torch.load(save_path, map_location=device)
    modifier.load_state_dict(state_dict)
    logger.info(f"\n+++ Load modifier.state_dict() from {save_path}\n")


class LinearM(nn.Module):
    def __init__(self, in_out_size, hook_module):
        super().__init__()
        self.hook_module = hook_module
        self.linear = nn.Linear(in_out_size, in_out_size)

    def forward(self, x):
        x = self.linear(x)
        return x


class MLPM(nn.Module):
    def __init__(self, in_out_size, hook_module):
        super().__init__()
        self.hook_module = hook_module
        self.up = nn.Linear(in_out_size, int(4 * in_out_size))
        self.down = nn.Linear(int(4 * in_out_size), in_out_size)

    def forward(self, x):
        x = self.up(x)
        x = F.gelu(x)
        x = self.down(x)
        return x


# generate using vanilla model
def generate_with_ori_model(model, tokenizer, input_ids_batch, max_new_tokens, verbose=False):
    with torch.no_grad():
        output_dict_ori = model.generate(input_ids_batch, max_new_tokens=max_new_tokens,
                                         return_dict_in_generate=True, output_logits=True)
        output = tokenizer.batch_decode(output_dict_ori.sequences, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)

        if verbose:
            logger.info(f"ori output: {output}")

    return output_dict_ori.logits, output


# generate using LKS-edited model
def generate_with_new_kb(model, tokenizer, modifier, batch, max_new_tokens, verbose=False):
    input_ids_batch = batch.get("prompt_ids_batch")
    entity_last_pos_batch = batch.get("entity_last_pos_batch")

    new_kb = modifier(batch.get("feature_hs_batch"))

    def act_add(output):
        if isinstance(output, tuple):
            if output[0].shape[1] == 1:
                return output
            for i, entity_last_pos in enumerate(entity_last_pos_batch):
                if entity_last_pos < output[0].shape[1]:
                    output[0][i, entity_last_pos] = new_kb[i]
                else:
                    raise ValueError(
                        "entity_last_pos >= output[0].shape[1]: %d %d" % (entity_last_pos, output[0].shape[1]))
        else:
            if output.shape[1] == 1:
                return output

            for i, entity_last_pos in enumerate(entity_last_pos_batch):
                if entity_last_pos < output.shape[1]:
                    output[i, entity_last_pos] = new_kb[i]
                else:
                    raise ValueError("entity_last_pos >= output.shape[1]: %d %d" % (entity_last_pos, output.shape[1]))

        return output

    with Trace(model, layer=modifier.hook_module, edit_output=act_add) as ret:
        output_dict = model.generate(input_ids_batch, max_new_tokens=max_new_tokens,
                                     return_dict_in_generate=True, output_logits=True)
        output = tokenizer.batch_decode(output_dict.sequences, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)

        if verbose:
            logger.info(f"mdf output: {output}")

    return output_dict.logits, output


def get_model_inputs(tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, padding="longest")
    inputs.to(device)

    return inputs


def get_output_tokens_batch(tokenizer, top_indices, desc=""):
    output = tokenizer.batch_decode(top_indices, skip_special_tokens=False)
    logger.info(f"{desc} output: {output}")

    return output


def get_top_token_ids(logits, batch):
    next_prompt_index_batch = batch.get("next_prompt_index_batch", [])
    target_len_batch = batch.get("target_len_batch", [])

    top_indices = []
    for i, (npi, t_len) in enumerate(zip(next_prompt_index_batch, target_len_batch)):
        top_indices_i = []
        for token_index in range(npi, npi + t_len):
            token_id = torch.topk(logits[i, token_index, :], 1, dim=-1).indices
            top_indices_i.extend(token_id.detach().cpu().numpy())
        top_indices.append(top_indices_i)

    return top_indices


# output using vanilla model
def output_with_ori_model(model, input_ids_batch):
    with torch.no_grad():
        output_dict_ori = model(input_ids_batch, return_dict=True)

    return output_dict_ori.logits


# output using LKS-edited model
def output_with_new_kb(model, modifier, batch):
    input_ids_batch = batch.get("prompt_ids_batch")
    entity_last_pos_batch = batch.get("entity_last_pos_batch")

    new_kb = modifier(batch.get("feature_hs_batch"))

    def act_add(output):
        if isinstance(output, tuple):
            for i, entity_last_pos in enumerate(entity_last_pos_batch):
                if entity_last_pos < output[0].shape[1]:
                    output[0][i, entity_last_pos] = new_kb[i]
                else:
                    raise ValueError(
                        "entity_last_pos >= output[0].shape[1]: %d %d" % (entity_last_pos, output[0].shape[1]))
        else:
            for i, entity_last_pos in enumerate(entity_last_pos_batch):
                if entity_last_pos < output.shape[1]:
                    output[i, entity_last_pos] = new_kb[i]
                else:
                    raise ValueError("entity_last_pos >= output.shape[1]: %d %d" % (entity_last_pos, output.shape[1]))

        return output

    module = modifier.hook_module
    with Trace(model, module, edit_output=act_add) as ret:
        output_dict = model(input_ids_batch, return_dict=True)

    return output_dict.logits


def get_loss_loc(model, tokenizer, batch, modifier, verbose):
    prompt_ids_batch = batch.get("prompt_ids_batch")
    next_prompt_index_batch = batch.get("next_prompt_index_batch")

    output_logits_ori = output_with_ori_model(model, prompt_ids_batch)
    output_logits = output_with_new_kb(model, modifier, batch)

    loss_loc = torch.tensor(0.).to(model.device)
    logits_softmax_ori = torch.log_softmax(output_logits_ori, dim=-1)
    logits_softmax = torch.log_softmax(output_logits, dim=-1)
    for bi, npi in enumerate(next_prompt_index_batch):
        loc_logits = logits_softmax[bi, npi:npi + 3]
        loc_logits_ori = logits_softmax_ori[bi, npi:npi + 3]
        loss_loc += nn.functional.kl_div(loc_logits, loc_logits_ori, reduction="sum", log_target=True)
    loss_loc /= len(prompt_ids_batch)

    return loss_loc


def get_loss_target(model, tokenizer, batch, modifier, verbose):
    if verbose:
        print("\n\ntarget_text_batch: ", batch.get("target_batch"))

    next_prompt_index_batch = batch.get("next_prompt_index_batch")
    target_ids_batch = batch.get("target_ids_batch", [])
    target_len_batch = batch.get("target_len_batch", [])

    output_logits_ori = output_with_ori_model(model, batch.get("prompt_ids_batch"))
    output_logits = output_with_new_kb(model, modifier, batch)
    if verbose:
        top_id_ori_batch = get_top_token_ids(output_logits_ori, batch)
        get_output_tokens_batch(tokenizer, top_id_ori_batch, desc="ori")
        top_id_batch = get_top_token_ids(output_logits, batch)
        get_output_tokens_batch(tokenizer, top_id_batch, desc="mdf")

    logits_softmax = torch.log_softmax(output_logits, dim=-1)
    loss_target = torch.tensor(0.).to(model.device)
    for i, (npi, t_len) in enumerate(zip(next_prompt_index_batch, target_len_batch)):
        loss_target_i = torch.tensor(0.).to(model.device)
        for j, token_index in enumerate(range(npi, npi + t_len)):
            loss_target_i += - logits_softmax[i, token_index, target_ids_batch[i][j]]
        loss_target += loss_target_i / t_len

    loss_target /= len(target_len_batch)

    if verbose:
        print("\n==> loss_target: ", loss_target, "\n")

    return loss_target


def train_modifier(modifier, model, tokenizer, datasets_dict, train_args, edit_scope, modifier_save_path):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    optimizer = optim.AdamW(modifier.parameters(), lr=train_args.lr)
    
    total_batch_num = (bool(train_args.p_edit) * len(datasets_dict.get("train_dataset")) +
                       bool(train_args.p_eq) * len(datasets_dict.get("rephrase_dataset")) +
                       bool(train_args.p_loc) * len(datasets_dict.get("loc_dataset")))
    num_training_steps = train_args.max_epoch * total_batch_num
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # train
    logger.info(f"*-----------train-----------*")
    patience = 3
    best_loss = float('inf')
    no_improve_epoch_num = 0
    # if train_args.data_num <= 10:
    #     train_args.max_epoch = min(train_args.max_epoch, 10)
    logger.info("max_epoch: %d", train_args.max_epoch)

    for epoch in range(train_args.max_epoch):
        train_loss = 0
        modifier.train()
        loss_edit = 0
        loss_eq = 0
        loss_loc = 0
        if train_args.p_edit:
            for batch in tqdm(datasets_dict.get("train_dataset")):
                new_batch = data_handler.prepare_batch_data_train(model, tokenizer, batch, modifier.hook_module,
                                                                  edit_scope)

                loss = train_args.p_edit * get_loss_target(model, tokenizer, new_batch, modifier, train_args.verbose)
                train_loss += loss.item()
                loss_edit += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            loss_edit /= len(datasets_dict.get("train_dataset"))
            logger.info(f"** epoch {epoch} - loss_edit: {loss_edit}")
        if train_args.p_eq:
            for batch in tqdm(datasets_dict.get("rephrase_dataset")):
                new_batch = data_handler.prepare_batch_data_train(model, tokenizer, batch, modifier.hook_module,
                                                                  edit_scope)

                loss = train_args.p_eq * get_loss_target(model, tokenizer, new_batch, modifier, train_args.verbose)
                train_loss += loss.item()
                loss_eq += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            loss_eq /= len(datasets_dict.get("rephrase_dataset"))
            logger.info(f"** epoch {epoch} - loss_eq: {loss_eq}")
        if train_args.p_loc:
            for batch in tqdm(datasets_dict.get("loc_dataset")):
                new_batch = data_handler.prepare_batch_data_train(model, tokenizer, batch, modifier.hook_module,
                                                                  edit_scope)
                loss = train_args.p_loc * get_loss_loc(model, tokenizer, new_batch, modifier, train_args.verbose)
                train_loss += loss.item()
                loss_loc += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            loss_loc /= len(datasets_dict.get("loc_dataset"))
            logger.info(f"** epoch {epoch} - loss_loc: {loss_loc}")

        train_loss /= total_batch_num
        logger.info(f"===> epoch {epoch} - loss: {train_loss}")

        # early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            no_improve_epoch_num = 0
            save_modifier(modifier.state_dict(), modifier_save_path)
        else:
            no_improve_epoch_num += 1
        if no_improve_epoch_num == patience or train_loss < 1e-2:
            logger.info("** Early stopping triggered")
            break

    return


def get_module_hidden_states_batch(model, inputs_batch, module):
    am_batch = inputs_batch.attention_mask
    last_token_index_batch = torch.tensor([sum(am) - 1 for am in am_batch])

    with torch.no_grad():
        with Trace(model, module, retain_output=True) as res:
            model(inputs_batch.input_ids)
            if isinstance(res.output, tuple):
                last_token_hidden_state = res.output[0][torch.arange(len(am_batch)), last_token_index_batch]
            else:
                last_token_hidden_state = res.output[torch.arange(len(am_batch)), last_token_index_batch]

    return last_token_hidden_state


def load_model(model_name, device, half_precision, local=True):
    if local:
        model_dict = {"llama2": LLAMA2_NAME_LOCAL,
                      "mistral": MISTRAL_NAME_LOCAL}
    else:
        model_dict = {"llama2": LLAMA2_NAME,
                      "mistral": MISTRAL_NAME}

    model_path = model_dict.get(model_name)
    if not model_path:
        logger.error("Wrong model name.")
        raise ValueError("Wrong model name.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    if half_precision:
        logger.info("Load model in torch.bfloat16")
        model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN,
                                                     torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN)

    model.eval()
    model.to(device)

    # calculate GPU usage
    memory_allocated_gb = torch.cuda.memory_allocated(device=device) / (1024 ** 3)
    logger.info(f"** Memory allocated by model: {memory_allocated_gb} GB\n")

    return model, tokenizer

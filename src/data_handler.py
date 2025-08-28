import json
import os
import shutil
import Levenshtein
import datasets
import spacy
import torch
from pathlib import Path
from fuzzywuzzy import fuzz, process
import model_handler

nlp = spacy.load("en_core_web_lg")
TEMP_DIR = "../temp/"


def load_json(json_path):
    with open(json_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)

    return data


def save_json(json_path, data):
    with open(json_path, "w", encoding='utf-8') as fp:  # a
        json.dump(data, fp, indent=4)


# obtain the position of entity token within prompt tokens
def get_entity_position(prompt_offset_mapping, prompt, entity):
    prompt_offset_mapping = prompt_offset_mapping.tolist()

    if entity in prompt:
        char_left = prompt.index(entity)
        char_right = char_left + len(entity) - 1
    else:
        raise ValueError("No entity in prompt. entity: %s; prompt: %s" % (entity, prompt))

    token_left, token_right = -1, -1

    for i, om_tuple in enumerate(prompt_offset_mapping):
        if token_left >= 0 and token_right >= 0:
            break

        if om_tuple[0] <= char_left < om_tuple[1]:
            token_left = i
        if om_tuple[0] <= char_right < om_tuple[1]:
            token_right = i

    return (token_left, token_right)


def get_entity_position_batch(prompt_offset_mapping_batch, prompt_batch, entity_batch):
    entity_position_batch = []

    for prompt_offset_mapping, prompt, entity in zip(prompt_offset_mapping_batch, prompt_batch, entity_batch):
        entity_position = get_entity_position(prompt_offset_mapping, prompt, entity)
        entity_position_batch.append(entity_position)

    return entity_position_batch


def get_prompt_data_batch(tokenizer, prompt_batch, device):
    prompt_inputs_batch = model_handler.get_model_inputs(tokenizer, prompt_batch, device)
    next_prompt_index_batch = torch.tensor([sum(am) - 1 for am in prompt_inputs_batch.attention_mask])

    return prompt_inputs_batch, next_prompt_index_batch


# get target and its token ids
def get_target_data_batch(tokenizer, target_batch, device):
    target_ids_batch = []
    target_len_batch = []
    for target in target_batch:
        target_ids = model_handler.get_model_inputs(tokenizer, target, device).input_ids.detach().squeeze(0)
        if target_ids[0] == 1 or target_ids[0] == 128000:
            target_ids = target_ids[1:]

        target_ids_batch.append(target_ids)
        target_len_batch.append(len(target_ids))

    return target_ids_batch, target_len_batch


def prepare_batch_data_train(model, tokenizer, data_batch, module, edit_scope):
    # entity data
    entity_batch = data_batch.get("subject", [])
    entity_inputs_batch = model_handler.get_model_inputs(tokenizer, entity_batch, model.device)
    entity_hs_batch = model_handler.get_module_hidden_states_batch(model, entity_inputs_batch, module)

    # prompt data
    prompt_batch = data_batch.get("prompt", [])
    prompt_inputs_batch, next_prompt_index_batch = get_prompt_data_batch(tokenizer, prompt_batch, model.device)

    # entity token index
    entity_pos_tuple_batch = get_entity_position_batch(prompt_inputs_batch.offset_mapping,
                                                       prompt_batch, entity_batch)
    entity_last_pos_batch = [entity_pos_tuple[1] for entity_pos_tuple in entity_pos_tuple_batch]

    # unaffected token index
    no_influ_pos_batch = [(elp + 1, npi) for elp, npi in zip(entity_last_pos_batch, next_prompt_index_batch)]

    # target data
    target_batch = data_batch.get("target", [])
    target_ids_batch, target_len_batch = get_target_data_batch(tokenizer, target_batch, model.device)

    # prompt + " " + target
    prompt_target_batch = [prompt + " " + target for prompt, target in zip(prompt_batch, target_batch)]
    prompt_target_ids_batch = model_handler.get_model_inputs(tokenizer, prompt_target_batch, model.device).input_ids

    # feature data
    feature_batch = [edit_scope.get(entity, "") for entity in entity_batch]
    feature_inputs_batch = model_handler.get_model_inputs(tokenizer, feature_batch, model.device)
    feature_hs_batch = model_handler.get_module_hidden_states_batch(model, feature_inputs_batch, module).detach()

    return {"entity_batch": entity_batch,
            "entity_hs_batch": entity_hs_batch,
            "entity_last_pos_batch": entity_last_pos_batch,
            "feature_hs_batch": feature_hs_batch,
            "target_batch": target_batch,
            "target_ids_batch": target_ids_batch,
            "target_len_batch": target_len_batch,
            "prompt_batch": prompt_batch,
            "prompt_ids_batch": prompt_target_ids_batch,
            "no_influ_pos_batch": no_influ_pos_batch,
            "next_prompt_index_batch": next_prompt_index_batch
            }


# detect the target entity in user input
def find_entity(entity_feature_dict, prompt):
    editor_entity_list = list(entity_feature_dict.keys())
    editor_entity_lower_list = [ent.lower().strip() for ent in editor_entity_list]

    entity_official = ""
    for editor_entity in editor_entity_list:
        if editor_entity in prompt:
            if len(editor_entity) > len(entity_official):
                entity_official = editor_entity
                entity_in_prompt = editor_entity
    if entity_official:
        return entity_in_prompt, entity_official

    prompt_doc = nlp(prompt)
    possible_ents_list = [ent.text for ent in prompt_doc.ents]
    if not possible_ents_list:
        possible_ents_list = [token.text for token in prompt_doc]

    # fuzz
    match_partial = process.extractOne(prompt.lower(), editor_entity_lower_list, scorer=fuzz.partial_ratio)
    match_token_sort = process.extractOne(prompt.lower(), editor_entity_lower_list, scorer=fuzz.token_sort_ratio)

    # Levenshtein
    if match_partial[1] <= 90 and match_token_sort[1] <= 50:
        candi_dict_l = {}
        for ent in possible_ents_list:
            ratio_list = [Levenshtein.ratio(ent, e) for e in editor_entity_list]
            ent_max_ratio = max(ratio_list)
            editor_entity = editor_entity_list[ratio_list.index(ent_max_ratio)]
            candi_dict_l.update({float(ent_max_ratio): [ent, editor_entity]})

        max_ratio = max(candi_dict_l.keys())
        entity_in_prompt = candi_dict_l.get(max_ratio)[0]
        entity_official = candi_dict_l.get(max_ratio)[1]

        if max_ratio > 0.64:
            return entity_in_prompt, entity_official

        return "", ""

    if match_partial[1] >= 90:
        if match_token_sort[1] > 60:
            entity_find = match_token_sort[0]
        else:
            entity_find = match_partial[0]
    else:
        entity_find = match_token_sort[0]

    entity_official = editor_entity_list[editor_entity_lower_list.index(entity_find)]

    if possible_ents_list:
        entity_in_prompt = process.extractOne(entity_official, possible_ents_list, scorer=fuzz.partial_ratio)[0]
    else:
        entity_in_prompt = prompt.replace("?", " ").replace(".", " ").strip().split(" ")[-1]

    return entity_in_prompt, entity_official


def get_possible_entity_feature(entity_feature_dict, prompt):
    feature = ""
    # get entity
    entity, entity_official = find_entity(entity_feature_dict, prompt)

    if entity:
        feature = entity_feature_dict.get(entity_official, "")

    return entity, feature


def prepare_data_eval(model, tokenizer, data_item, module, edit_scope):
    prompt = data_item.get("prompt", "")
    entity, feature = get_possible_entity_feature(edit_scope, prompt)
    prompt_inputs_batch, next_prompt_index_batch = get_prompt_data_batch(tokenizer, [prompt], model.device)
    target = data_item.get("target", "")
    target_ids_batch, target_len_batch = get_target_data_batch(tokenizer, [target], model.device)
    prompt_target = [prompt + " " + target]
    prompt_target_ids = model_handler.get_model_inputs(tokenizer, prompt_target, model.device).input_ids

    new_data_item = {"prompt_ids_batch": prompt_target_ids,
                     "target_batch": target,
                     "target_ids_batch": target_ids_batch,
                     "target_len_batch": target_len_batch,
                     "next_prompt_index_batch": next_prompt_index_batch
                     }

    if not feature:
        return new_data_item

    entity_pos_tuple_batch = get_entity_position_batch(prompt_inputs_batch.offset_mapping,
                                                       [prompt], [entity])
    entity_last_pos_batch = [entity_pos_tuple[1] for entity_pos_tuple in entity_pos_tuple_batch]
    feature_inputs_batch = model_handler.get_model_inputs(tokenizer, [feature], model.device)
    feature_hs_batch = model_handler.get_module_hidden_states_batch(model, feature_inputs_batch, module)

    new_data_item.update({"entity_last_pos_batch": entity_last_pos_batch,
                          "feature_hs_batch": feature_hs_batch})

    return new_data_item


def prepare_data_generation(model, tokenizer, data_item, module, edit_scope):
    prompt = data_item.get("prompt", "")
    entity, feature = get_possible_entity_feature(edit_scope, prompt)
    prompt_inputs_batch = model_handler.get_model_inputs(tokenizer, [prompt], model.device)
    data_item.update({"prompt_ids_batch": prompt_inputs_batch.input_ids})

    if not feature:
        return data_item

    entity_pos_tuple_batch = get_entity_position_batch(prompt_inputs_batch.offset_mapping,
                                                       [prompt], [entity])
    entity_last_pos_batch = [entity_pos_tuple[1] for entity_pos_tuple in entity_pos_tuple_batch]
    feature_inputs_batch = model_handler.get_model_inputs(tokenizer, [feature], model.device)
    feature_hs_batch = model_handler.get_module_hidden_states_batch(model, feature_inputs_batch, module)

    data_item.update({"entity_last_pos_batch": entity_last_pos_batch,
                      "feature_hs_batch": feature_hs_batch})

    return data_item


# generate edit scope dict
def generate_edit_scope(dataset, save_dir):
    new_data_dict = {}
    for i, data_dict in enumerate(dataset):
        subject = data_dict.get("subject", "").strip()
        target = data_dict.get("target", "").strip()
        prompt = data_dict.get("prompt", "").strip()
        feature = prompt + " " + target

        if new_data_dict.get(subject, ""):
            new_data_dict[subject] = new_data_dict.get(subject, "") + feature
        else:
            new_data_dict.update({subject: feature})

    save_path = Path(save_dir, "edit_scope.json")
    save_json(save_path, new_data_dict)
    print("Edit-scope saved at ", save_path)

    return new_data_dict


def build_train_dataset(dataset, batch_size, loc_num):
    if not dataset:
        raise ValueError("No training dataset.")

    total_loc_num = len(dataset[0].get("locality", []))
    if loc_num < 1:
        loc_num = total_loc_num
    else:
        loc_num = int(min(loc_num, total_loc_num))

    train_data_list = []
    rephrase_data_list = []
    loc_data_list = []
    for item_dict in dataset:
        train_data_list.append({"subject": item_dict.get("subject", ""),
                                "prompt": item_dict.get("prompt", ""),
                                "target": item_dict.get("target", "")})
        rephrase_data_list.append({"subject": item_dict.get("subject", ""),
                                   "prompt": item_dict.get("rephrase_prompt", ""),
                                   "target": item_dict.get("target", "")})
        for loc_p in item_dict.get("locality", [])[:loc_num]:
            loc_data_list.append({"subject": item_dict.get("subject", ""),
                                  "prompt": loc_p,
                                  "target": "here is the target position"})

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    train_temp_path = Path(TEMP_DIR, "train_data_list.json")
    rephrase_temp_path = Path(TEMP_DIR, "rephrase_data_list.json")
    loc_temp_path = Path(TEMP_DIR, "loc_data_list.json")
    save_json(train_temp_path, train_data_list)
    save_json(rephrase_temp_path, rephrase_data_list)
    save_json(loc_temp_path, loc_data_list)

    train_dataset = datasets.load_dataset("json", data_files=str(train_temp_path), split="train")
    rephrase_dataset = datasets.load_dataset("json", data_files=str(rephrase_temp_path), split="train")
    loc_dataset = datasets.load_dataset("json", data_files=str(loc_temp_path), split="train")

    train_dataset = train_dataset.batch(batch_size)
    rephrase_dataset = rephrase_dataset.batch(batch_size)
    loc_dataset = loc_dataset.batch(batch_size)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    return {"train_dataset": train_dataset,
            "rephrase_dataset": rephrase_dataset,
            "loc_dataset": loc_dataset
            }

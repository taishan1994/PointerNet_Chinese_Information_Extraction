import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self,
                 file_path=None,
                 data=None,
                 tokenizer=None,
                 max_len=None,
                 relation_label=None,
                 entity_label=None,
                 **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path, tokenizer, max_len, relation_label, entity_label)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path, tokenizer, max_len, relation_label, entity_label):
        return file_path


# 加载实体识别数据集
class ReDataset(ListDataset):
    @staticmethod
    def load_data(filename, tokenizer, max_len, relation_label, entity_label=None):

        data = []
        rel_label2id = {label: i for i, label in enumerate(relation_label)}
        ent_label2id = {label: i for i, label in enumerate(entity_label)}
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            f = json.loads(f)
            for d in f:
                tokens = d['tokens']
                text = "".join(tokens)

                if len(text) == 0:
                    continue
                entities = d["entities"]
                relations = d["relations"]

                sbj_tokens = tokens
                ner_tokens = tokens

                if len(sbj_tokens) > max_len - 2:
                    sbj_tokens = sbj_tokens[:max_len - 2]
                    ner_tokens = ner_tokens[:max_len - 2]

                sbj_tokens = ['[CLS]'] + sbj_tokens + ['[SEP]']
                sbj_start_label = [0] * len(sbj_tokens)
                sbj_end_label = [0] * len(sbj_tokens)

                ner_tokens = ['[CLS]'] + ner_tokens + ['[SEP]']
                ner_start_label = np.zeros((len(entity_label), max_len))
                ner_end_label = np.zeros((len(entity_label), max_len))

                for ent in entities:
                    e_type = ent["type"]
                    e_start = ent["start"]
                    e_end = ent["end"]
                    sbj_start_label[e_start + 1] = 1
                    sbj_end_label[e_end] = 1

                    ner_start_label[ent_label2id[e_type]][e_start + 1] = 1
                    ner_end_label[ent_label2id[e_type]][e_end] = 1

                sbj_data = {
                    "sbj_tokens": sbj_tokens,
                    "sbj_start_labels": sbj_start_label,
                    "sbj_end_labels": sbj_end_label,
                }

                ner_data = {
                    "ner_tokens": ner_tokens,
                    "ner_start_labels": ner_start_label,
                    "ner_end_labels": ner_end_label,
                }

                other_data = []

                for rel in relations:
                    sbj_ent_id = rel["head"]
                    obj_ent_id = rel["tail"]
                    rel_type = rel["type"]

                    sbj_ent = entities[sbj_ent_id]

                    if sbj_ent["end"] >= max_len - 1:
                        continue

                    sbj_ent_text = text[sbj_ent["start"]:sbj_ent["end"]]

                    pre = [i for i in sbj_ent_text] + ['[SEP]']
                    obj_tokens = pre + tokens
                    if len(obj_tokens) > max_len - 2:
                        obj_tokens = obj_tokens[:max_len - 2]
                    obj_tokens = ['[CLS]'] + obj_tokens + ['[SEP]']

                    obj_ent = entities[obj_ent_id]
                    obj_ent_end = obj_ent["end"]
                    obj_ent_text = text[obj_ent["start"]:obj_ent["end"]]

                    rel_tokens = pre + [i for i in obj_ent_text] + ['[SEP]']
                    if obj_ent_end + len(rel_tokens) >= max_len - 1:
                        continue

                    if len(rel_tokens + tokens) > max_len - 2:
                        tmp_tokens = (rel_tokens + tokens)[:max_len - 2]
                    else:
                        tmp_tokens = rel_tokens + tokens

                    rel_tokens = ['[CLS]'] + tmp_tokens + ['[SEP]']

                    obj_start_label = [0] * len(obj_tokens)
                    obj_end_label = [0] * len(obj_tokens)

                    obj_start_label[obj_ent["start"] + 1 + len(pre)] = 1
                    obj_end_label[obj_ent["end"] + len(pre)] = 1

                    rel_label = [0] * len(relation_label)
                    rel_label[rel_label2id[rel_type]] = 1

                    assert len(rel_tokens) <= max_len

                    other_data.append(
                        {
                            "obj_tokens": obj_tokens,
                            "obj_start_labels": obj_start_label,
                            "obj_end_labels": obj_end_label,
                            "rel_tokens": rel_tokens,
                            "rel_label": rel_label,
                        }
                    )

                data.append(
                    {
                        "sbj_data": sbj_data,
                        "ner_data": ner_data,
                        "other_data": other_data,
                    }
                )

        return data


def convert_list_to_tensor(alist, dtype=torch.long):
    return torch.tensor(np.array(alist) if isinstance(alist, list) else alist, dtype=dtype)


class ReCollate:
    def __init__(self,
                 max_len,
                 tokenizer,
                 tasks):
        self.maxlen = max_len
        self.tokenizer = tokenizer
        self.tasks = tasks

    def collate_fn(self, batch):
        batch_ner_token_ids = []
        batch_ner_attention_mask = []
        batch_ner_token_type_ids = []
        batch_ner_start_labels = []
        batch_ner_end_labels = []
        batch_sbj_token_ids = []
        batch_sbj_attention_mask = []
        batch_sbj_token_type_ids = []
        batch_sbj_start_labels = []
        batch_sbj_end_labels = []
        batch_obj_token_ids = []
        batch_obj_attention_mask = []
        batch_obj_token_type_ids = []
        batch_obj_start_labels = []
        batch_obj_end_labels = []
        batch_rel_token_ids = []
        batch_rel_attention_mask = []
        batch_rel_token_type_ids = []
        batch_rel_labels = []
        for i, data in enumerate(batch):
            ner_data = data["ner_data"]
            sbj_data = data["sbj_data"]
            other_data = data["other_data"]

            sbj_token_type_ids = [0] * self.maxlen
            ner_token_type_ids = [0] * self.maxlen
            obj_token_type_ids = [0] * self.maxlen
            rel_token_type_ids = [0] * self.maxlen

            sbj_tokens = sbj_data["sbj_tokens"]
            sbj_tokens = self.tokenizer.convert_tokens_to_ids(sbj_tokens)
            sbj_start_labels = sbj_data["sbj_start_labels"]
            sbj_end_labels = sbj_data["sbj_end_labels"]

            ner_tokens = ner_data["ner_tokens"]
            ner_tokens = self.tokenizer.convert_tokens_to_ids(ner_tokens)
            ner_start_labels = ner_data["ner_start_labels"]
            ner_end_labels = ner_data["ner_end_labels"]

            if len(sbj_tokens) < self.maxlen:
                sbj_attention_mask = [1] * len(sbj_tokens) + [0] * (self.maxlen - len(sbj_tokens))
                sbj_start_labels = sbj_start_labels + [0] * (self.maxlen - len(sbj_tokens))
                sbj_end_labels = sbj_end_labels + [0] * (self.maxlen - len(sbj_tokens))
                sbj_tokens = sbj_tokens + [0] * (self.maxlen - len(sbj_tokens))
            else:
                sbj_attention_mask = [1] * self.maxlen

            if len(ner_tokens) < self.maxlen:
                ner_attention_mask = [1] * len(ner_tokens) + [0] * (self.maxlen - len(ner_tokens))
                ner_tokens = ner_tokens + [0] * (self.maxlen - len(ner_tokens))
            else:
                ner_attention_mask = [1] * self.maxlen

            for odata in other_data:
                obj_tokens = odata["obj_tokens"]
                obj_tokens = self.tokenizer.convert_tokens_to_ids(obj_tokens)
                obj_start_labels = odata["obj_start_labels"]
                obj_end_labels = odata["obj_end_labels"]
                rel_tokens = odata["rel_tokens"]
                rel_tokens = self.tokenizer.convert_tokens_to_ids(rel_tokens)
                rel_label = odata["rel_label"]

                if len(obj_tokens) < self.maxlen:
                    obj_start_labels = obj_start_labels + [0] * (self.maxlen - len(obj_tokens))
                    obj_end_labels = obj_end_labels + [0] * (self.maxlen - len(obj_tokens))
                    obj_attention_mask = [1] * len(obj_tokens) + [0] * (self.maxlen - len(obj_tokens))
                    obj_tokens = obj_tokens + [0] * (self.maxlen - len(obj_tokens))
                else:
                    obj_attention_mask = [1] * self.maxlen

                if len(rel_tokens) < self.maxlen:
                    rel_attention_mask = [1] * len(rel_tokens) + [0] * (self.maxlen - len(rel_tokens))
                    rel_tokens = rel_tokens + [0] * (self.maxlen - len(rel_tokens))
                else:
                    rel_attention_mask = [1] * self.maxlen

                batch_obj_token_ids.append(obj_tokens)
                batch_obj_attention_mask.append(obj_attention_mask)
                batch_obj_token_type_ids.append(obj_token_type_ids)
                batch_obj_start_labels.append(obj_start_labels)
                batch_obj_end_labels.append(obj_end_labels)

                batch_rel_token_ids.append(rel_tokens)
                batch_rel_attention_mask.append(rel_attention_mask)
                batch_rel_token_type_ids.append(rel_token_type_ids)
                batch_rel_labels.append(rel_label)

                assert len(obj_tokens) == self.maxlen
                assert len(obj_attention_mask) == self.maxlen
                assert len(rel_tokens) == self.maxlen
                assert len(rel_attention_mask) == self.maxlen

            assert len(sbj_tokens) == self.maxlen
            assert len(sbj_attention_mask) == self.maxlen
            assert len(ner_tokens) == self.maxlen
            assert len(ner_attention_mask) == self.maxlen

            batch_sbj_token_ids.append(sbj_tokens)
            batch_sbj_attention_mask.append(sbj_attention_mask)
            batch_sbj_token_type_ids.append(sbj_token_type_ids)
            batch_sbj_start_labels.append(sbj_start_labels)
            batch_sbj_end_labels.append(sbj_end_labels)

            batch_ner_token_ids.append(ner_tokens)
            batch_ner_attention_mask.append(ner_attention_mask)
            batch_ner_token_type_ids.append(ner_token_type_ids)
            batch_ner_start_labels.append(ner_start_labels)
            batch_ner_end_labels.append(ner_end_labels)

        batch_ner_token_ids = convert_list_to_tensor(batch_ner_token_ids)
        batch_ner_attention_mask = convert_list_to_tensor(batch_ner_attention_mask)
        batch_ner_token_type_ids = convert_list_to_tensor(batch_ner_token_type_ids)
        batch_ner_start_labels = convert_list_to_tensor(batch_ner_start_labels, dtype=torch.float)
        batch_ner_end_labels = convert_list_to_tensor(batch_ner_end_labels, dtype=torch.float)

        batch_sbj_token_ids = convert_list_to_tensor(batch_sbj_token_ids)
        batch_sbj_attention_mask = convert_list_to_tensor(batch_sbj_attention_mask)
        batch_sbj_token_type_ids = convert_list_to_tensor(batch_sbj_token_type_ids)
        batch_sbj_start_labels = convert_list_to_tensor(batch_sbj_start_labels, dtype=torch.float)
        batch_sbj_end_labels = convert_list_to_tensor(batch_sbj_end_labels, dtype=torch.float)

        batch_obj_token_ids = convert_list_to_tensor(batch_obj_token_ids)
        batch_obj_attention_mask = convert_list_to_tensor(batch_obj_attention_mask)
        batch_obj_token_type_ids = convert_list_to_tensor(batch_obj_token_type_ids)
        batch_obj_start_labels = convert_list_to_tensor(batch_obj_start_labels, dtype=torch.float)
        batch_obj_end_labels = convert_list_to_tensor(batch_obj_end_labels, dtype=torch.float)

        batch_rel_token_ids = convert_list_to_tensor(batch_rel_token_ids)
        batch_rel_attention_mask = convert_list_to_tensor(batch_rel_attention_mask)
        batch_rel_token_type_ids = convert_list_to_tensor(batch_rel_token_type_ids)
        batch_rel_labels = convert_list_to_tensor(batch_rel_labels, dtype=torch.float)

        ner_res = {
            "ner_input_ids": batch_ner_token_ids,
            "ner_attention_mask": batch_ner_attention_mask,
            "ner_token_type_ids": batch_ner_token_type_ids,
            "ner_start_labels": batch_ner_start_labels,
            "ner_end_labels": batch_ner_end_labels,
        }

        sbj_res = {
            "re_sbj_input_ids": batch_sbj_token_ids,
            "re_sbj_attention_mask": batch_sbj_attention_mask,
            "re_sbj_token_type_ids": batch_sbj_token_type_ids,
            "re_sbj_start_labels": batch_sbj_start_labels,
            "re_sbj_end_labels": batch_sbj_end_labels,
        }

        sbj_obj_res = {
            "re_obj_input_ids": batch_obj_token_ids,
            "re_obj_attention_mask": batch_obj_attention_mask,
            "re_obj_token_type_ids": batch_obj_token_type_ids,
            "re_obj_start_labels": batch_obj_start_labels,
            "re_obj_end_labels": batch_obj_end_labels,
        }

        sbj_obj_rel_res = {
            "re_rel_input_ids": batch_rel_token_ids,
            "re_rel_attention_mask": batch_rel_attention_mask,
            "re_rel_token_type_ids": batch_rel_token_type_ids,
            "re_rel_labels": batch_rel_labels,
        }

        if "ner" in self.tasks:
            return ner_res
        elif "sbj" in self.tasks:
            return sbj_res
        elif "obj" in self.tasks:
            return sbj_obj_res
        elif "rel" in self.tasks:
            return sbj_obj_rel_res
        else:
            raise Exception("请至少选择一项任务！")


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext')

    # 测试实体识别
    # ============================
    max_seq_len = 256
    with open("data/re/ske/entity_labels.txt", "r", encoding="utf-8") as fp:
        entity_label = fp.read().strip().split("\n")

    with open("data/re/ske/relation_labels.txt", "r", encoding="utf-8") as fp:
        relation_label = fp.read().strip().split("\n")

    print(entity_label)
    print(relation_label)
    train_dataset = ReDataset(file_path='data/re/ske/train.json',
                              tokenizer=tokenizer,
                              max_len=max_seq_len,
                              relation_label=relation_label,
                              entity_label=entity_label)

    print(train_dataset[0])
    # for k, v in train_dataset[0].items():
    #     print(k, v)

    collate = ReCollate(max_len=max_seq_len, tokenizer=tokenizer, tasks=["ner"])
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

    for i, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            print(k, v.shape)
        break

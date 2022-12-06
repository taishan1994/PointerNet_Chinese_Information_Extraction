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
                 entity_label=None,
                 tasks=None,
                 **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path, tokenizer, max_len, entity_label, tasks)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path, tokenizer, max_len, entity_label, tasks):
        return file_path


# 加载实体识别数据集
class EeDataset(ListDataset):
    @staticmethod
    def load_data(filename, tokenizer, max_len, entity_label=None, tasks=None):
        ner_data = []
        obj_data = []
        ent_label2id = {label: i for i, label in enumerate(entity_label)}
        with open(filename, encoding='utf-8') as f:
            f = f.read().strip().split("\n")
            for d in f:
                d = json.loads(d)
                text = d["text"]
                event_list = d["event_list"]
                if len(text) == 0:
                    continue
                
                if "ner" in tasks:
                  event_start_labels = np.zeros((len(ent_label2id), max_len))
                  event_end_labels = np.zeros((len(ent_label2id), max_len))

                  for event in event_list:
                    event_type = event["event_type"]
                    trigger = event["trigger"]
                    trigger_start_index = event["trigger_start_index"]
                    arguments = event["arguments"]

                    event_tokens = [i for i in text]

                    if len(event_tokens) > max_len - 2:
                        event_tokens = event_tokens[:max_len - 2]

                    event_tokens = ['[CLS]'] + event_tokens + ['[SEP]']
                    

                    if trigger_start_index+len(trigger) >= max_len - 1:
                      continue

                    event_start_labels[ent_label2id[event_type]][trigger_start_index+1] = 1
                    event_end_labels[ent_label2id[event_type]][trigger_start_index+len(trigger)] = 1
                  
                  
                  event_data = {
                      "ner_tokens": event_tokens,
                      "ner_start_labels": event_start_labels,
                      "ner_end_labels": event_end_labels,
                  }
                  ner_data.append(event_data)
                elif "obj" in tasks:
                  for event in event_list:
                    event_type = event["event_type"]
                    trigger = event["trigger"]
                    trigger_start_index = event["trigger_start_index"]
                    arguments = event["arguments"]
                    for argument in arguments:
                      argument_start_index = argument["argument_start_index"]
                      role = argument["role"]
                      pre_tokens = [i for i in event_type + "_" + role] + ['[SEP]']
                      argu = argument["argument"]
                      if len(text) + len(pre_tokens) > max_len - 2:
                        argu_token = (pre_tokens + [i for i in text])[:max_len-2]
                      else:
                        argu_token = pre_tokens + [i for i in text]
                      argu_token = ['[CLS]'] + argu_token + ['[SEP]']
                      argu_start_labels = [0] * len(argu_token)
                      argu_end_labels = [0] * len(argu_token)
                      argu_start = len(pre_tokens) + 1 + argument_start_index
                      argu_end = argu_start + len(argu) - 1
                      if argu_end >= max_len - 1:
                        continue
                      argu_start_labels[argu_start] = 1
                      argu_end_labels[argu_end] = 1
                     
                      argu_data = {
                        "obj_tokens": argu_token,
                        "obj_start_labels": argu_start_labels,
                        "obj_end_labels": argu_end_labels,
                      }
  
                      obj_data.append(argu_data)

        return ner_data if "ner" in tasks else obj_data


def convert_list_to_tensor(alist, dtype=torch.long):
    return torch.tensor(np.array(alist) if isinstance(alist, list) else alist, dtype=dtype)


class EeCollate:
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
        batch_obj_token_ids = []
        batch_obj_attention_mask = []
        batch_obj_token_type_ids = []
        batch_obj_start_labels = []
        batch_obj_end_labels = []
        for i, data in enumerate(batch):

            if "ner" in self.tasks:
              ner_token_type_ids = [0] * self.maxlen
              ner_tokens = data["ner_tokens"]
              ner_tokens = self.tokenizer.convert_tokens_to_ids(ner_tokens)
              ner_start_labels = data["ner_start_labels"]
              ner_end_labels = data["ner_end_labels"]

              if len(ner_tokens) < self.maxlen:
                ner_attention_mask = [1] * len(ner_tokens) + [0] * (self.maxlen - len(ner_tokens))
                ner_tokens = ner_tokens + [0] * (self.maxlen - len(ner_tokens))
              else:
                ner_attention_mask = [1] * self.maxlen

              batch_ner_token_ids.append(ner_tokens)
              batch_ner_attention_mask.append(ner_attention_mask)
              batch_ner_token_type_ids.append(ner_token_type_ids)
              batch_ner_start_labels.append(ner_start_labels)
              batch_ner_end_labels.append(ner_end_labels)


            elif "obj" in self.tasks:
              obj_tokens = data["obj_tokens"]
              obj_tokens = self.tokenizer.convert_tokens_to_ids(obj_tokens)
              obj_start_labels = data["obj_start_labels"]
              obj_end_labels = data["obj_end_labels"]
              obj_token_type_ids = [0] * self.maxlen

              if len(obj_tokens) < self.maxlen:
                    obj_start_labels = obj_start_labels + [0] * (self.maxlen - len(obj_tokens))
                    obj_end_labels = obj_end_labels + [0] * (self.maxlen - len(obj_tokens))
                    obj_attention_mask = [1] * len(obj_tokens) + [0] * (self.maxlen - len(obj_tokens))
                    obj_tokens = obj_tokens + [0] * (self.maxlen - len(obj_tokens))
              else:
                  obj_attention_mask = [1] * self.maxlen
            

              batch_obj_token_ids.append(obj_tokens)
              batch_obj_attention_mask.append(obj_attention_mask)
              batch_obj_token_type_ids.append(obj_token_type_ids)
              batch_obj_start_labels.append(obj_start_labels)
              batch_obj_end_labels.append(obj_end_labels)
      

        res = {}

        if "ner" in self.tasks:
          batch_ner_token_ids = convert_list_to_tensor(batch_ner_token_ids)
          batch_ner_attention_mask = convert_list_to_tensor(batch_ner_attention_mask)
          batch_ner_token_type_ids = convert_list_to_tensor(batch_ner_token_type_ids)
          batch_ner_start_labels = convert_list_to_tensor(batch_ner_start_labels, dtype=torch.float)
          batch_ner_end_labels = convert_list_to_tensor(batch_ner_end_labels, dtype=torch.float)
          ner_res = {
              "ner_input_ids": batch_ner_token_ids,
              "ner_attention_mask": batch_ner_attention_mask,
              "ner_token_type_ids": batch_ner_token_type_ids,
              "ner_start_labels": batch_ner_start_labels,
              "ner_end_labels": batch_ner_end_labels,
          }

          res = ner_res

        elif "obj" in self.tasks:
          batch_obj_token_ids = convert_list_to_tensor(batch_obj_token_ids)
          batch_obj_attention_mask = convert_list_to_tensor(batch_obj_attention_mask)
          batch_obj_token_type_ids = convert_list_to_tensor(batch_obj_token_type_ids)
          batch_obj_start_labels = convert_list_to_tensor(batch_obj_start_labels, dtype=torch.float)
          batch_obj_end_labels = convert_list_to_tensor(batch_obj_end_labels, dtype=torch.float)
          sbj_obj_res = {
              "re_obj_input_ids": batch_obj_token_ids,
              "re_obj_attention_mask": batch_obj_attention_mask,
              "re_obj_token_type_ids": batch_obj_token_type_ids,
              "re_obj_start_labels": batch_obj_start_labels,
              "re_obj_end_labels": batch_obj_end_labels,
          }

          res = sbj_obj_res
        
        return res





if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext')

    # 测试实体识别
    # ============================
    max_seq_len = 256
    with open("data/ee/duee/labels.txt", "r", encoding="utf-8") as fp:
        entity_label = fp.read().strip().split("\n")


    print(entity_label)
    tasks = ["ner"]
    train_dataset = EeDataset(file_path='data/ee/duee/duee_train.json',
                              tokenizer=tokenizer,
                              max_len=max_seq_len,
                              entity_label=entity_label,
                              tasks=tasks)

    print(len(train_dataset))
    # for k, v in train_dataset[0].items():
    #     print(k, v)

    collate = EeCollate(max_len=max_seq_len, tokenizer=tokenizer, tasks=tasks)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

    for i, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            print(k, v.shape)
        break

import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, tokenizer=None, max_len=None, label_list=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path, tokenizer, max_len, label_list)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path, tokenizer, max_len, label_list):
        return file_path


# 加载实体识别数据集
class NerDataset(ListDataset):
    @staticmethod
    def load_data(filename, tokenizer, max_len, label_list):
        data = []
        callback_info = []  # 用于计算评价指标
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            f = json.loads(f)
            for d in f:
                text = d['text']
                if len(text) == 0:
                    continue
                labels = d['labels']
                tokens = [i for i in text]
                if len(tokens) > max_len - 2:
                    tokens = tokens[:max_len - 2]
                    text = text[:max_len]
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                label = []
                label_dict = {x: [] for x in label_list}
                for lab in labels:  # 这里需要加上CLS的位置, lab[3]不用加1，因为是实体结尾的后一位
                    label.append([lab[2] + 1, lab[3], lab[1]])
                    label_dict.get(lab[1], []).append((text[lab[2]:lab[3]], lab[2]))
                data.append((token_ids, label))  # label为[[start, end, entity], ...]
                callback_info.append((text, label_dict))
        return data, callback_info


def convert_list_to_tensor(alist, dtype=torch.long):
    return torch.tensor(np.array(alist) if isinstance(alist, list) else alist, dtype=dtype)


class NerCollate:
    def __init__(self, max_len, label2id):
        self.maxlen = max_len
        self.label2id = label2id

    def collate_fn(self, batch):
        batch_token_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_start_labels = []
        batch_end_labels = []
        for i, (token_ids, text_labels) in enumerate(batch):
            start_labels = np.zeros((len(self.label2id), self.maxlen), dtype=np.long)
            end_labels = np.zeros((len(self.label2id), self.maxlen), dtype=np.long)
            token_type_ids = [0] * self.maxlen
            if len(token_ids) < self.maxlen:
                attention_mask = [1] * len(token_ids) + [0] * (self.maxlen - len(token_ids))
                token_ids = token_ids + [0] * (self.maxlen - len(token_ids))
            else:
                attention_mask = [1] * self.maxlen

            assert len(attention_mask) == self.maxlen
            assert len(token_type_ids) == self.maxlen
            assert len(token_ids) == self.maxlen
            batch_token_ids.append(token_ids)  # 前面已经限制了长度
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            for start, end, label in text_labels:
                # 排除SEP及之后的
                if end >= self.maxlen - 1:
                    continue
                label_id = self.label2id[label]
                start_labels[label_id][start] = 1
                end_labels[label_id][end] = 1
            batch_start_labels.append(start_labels)
            batch_end_labels.append(end_labels)
        batch_token_ids = convert_list_to_tensor(batch_token_ids)
        batch_token_type_ids = convert_list_to_tensor(batch_token_type_ids)
        batch_attention_mask = convert_list_to_tensor(batch_attention_mask)
        batch_start_labels = convert_list_to_tensor(batch_start_labels, dtype=torch.float)
        batch_end_labels = convert_list_to_tensor(batch_end_labels, dtype=torch.float)
        res = {
            "input_ids": batch_token_ids,
            "token_type_ids": batch_token_type_ids,
            "attention_mask": batch_attention_mask,
            "ner_start_labels": batch_start_labels,
            "ner_end_labels": batch_end_labels,
        }
        return res




if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('../chinese-bert-wwm-ext')

    # 测试实体识别
    # ============================
    max_seq_len = 150
    with open("data/ner/cner/labels.txt","r") as fp:
        labels = fp.read().strip().split("\n")
    train_dataset, train_callback = NerDataset(file_path='data/ner/cner/test.json',
                                               tokenizer=tokenizer,
                                               max_len=max_seq_len,
                                               label_list=labels)
    print(train_dataset[0])
    print(train_callback[0])

    id2tag = {}
    tag2id = {}
    for i, label in enumerate(labels):
        id2tag[i] = label
        tag2id[label] = i

    collate = NerCollate(max_len=max_seq_len, tag2id=tag2id)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

    for i, batch in enumerate(train_dataloader):
        for k, v in batch.items():
            print(k,v.shape)
    # ============================

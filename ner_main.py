import sys

sys.path.append('..')
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from UIE.model import UIEModel
from UIE.config import NerArgs
from UIE.ner_data_loader import NerDataset, NerCollate
from UIE.utils.decode import ner_decode
from UIE.utils.metrics import calculate_metric, classification_report, get_p_r_f


class NerPipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir, map_location="cpu"))

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if "bert" in space[0]:
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    def eval_forward(self, data_loader):
        s_logits, e_logits = [], []
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(self.args.device)
            output = model(batch_data['input_ids'],
                           batch_data['token_type_ids'],
                           batch_data['attention_mask'],
                           )
            start_logits = output["ner_output"]["ner_start_logits"]
            end_logits = output["ner_output"]["ner_end_logits"]

            batch_size = batch_data['input_ids'].size(0)
            for i in range(batch_size):
              s_logits.append([logit[i, :] for logit in start_logits])
              e_logits.append([logit[i, :] for logit in end_logits])


        return s_logits, e_logits

    def get_metric(self, s_logits, e_logits, callback):
        batch_size = len(callback)
        total_count = [0 for _ in range(len(self.args.id2label))]
        role_metric = np.zeros([len(self.args.id2label), 3])
        for s_logit, e_logit, tmp_callback in zip(s_logits, e_logits, callback):
          text, gt_entities = tmp_callback
          pred_entities = ner_decode(s_logit, e_logit, text, self.args.id2label)
          # print("========================")
          # print(pred_entities)
          # print(gt_entities)
          # print("========================")
          for idx, _type in enumerate(self.args.labels):
              if _type not in pred_entities:
                  pred_entities[_type] = []
              total_count[idx] += len(gt_entities[_type])
              role_metric[idx] += calculate_metric(pred_entities[_type], gt_entities[_type])
                                                  
        return role_metric, total_count

    def train(self, dev=True):
        train_dataset, train_callback = NerDataset(file_path=self.args.train_path,
                                                   tokenizer=self.args.tokenizer,
                                                   max_len=self.args.max_seq_len,
                                                   label_list=self.args.labels)
        collate = NerCollate(max_len=self.args.max_seq_len, label2id=self.args.label2id)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2,
                                  collate_fn=collate.collate_fn)
        dev_loader = None
        dev_callback = None
        if dev:
            dev_dataset, dev_callback = NerDataset(file_path=self.args.dev_path,
                                                   tokenizer=self.args.tokenizer,
                                                   max_len=self.args.max_seq_len,
                                                   label_list=self.args.labels)
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.args.eval_batch_size,
                                    shuffle=False,
                                    num_workers=2,
                                    collate_fn=collate.collate_fn)

        t_total = len(train_loader) * self.args.train_epoch
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)

        global_step = 0
        self.model.zero_grad()
        self.model.to(self.args.device)
        eval_step = self.args.eval_step
        best_f1 = 0.
        for epoch in range(1, self.args.train_epoch + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.args.device)
                output = self.model(batch_data['input_ids'],
                                    batch_data['token_type_ids'],
                                    batch_data['attention_mask'],
                                    batch_data['ner_start_labels'],
                                    batch_data['ner_end_labels'])
                loss = output["ner_output"]["ner_loss"]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print('【train】 Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))
                if dev and global_step % eval_step == 0:
                    s_logits, e_logits = self.eval_forward(dev_loader)
                    role_metric, _ = self.get_metric(s_logits, e_logits, dev_callback)
                    mirco_metrics = np.sum(role_metric, axis=0)
                    mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                    print('【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0],
                                                                                         mirco_metrics[1],
                                                                                         mirco_metrics[2]))
                    if mirco_metrics[2] > best_f1:
                        best_f1 = mirco_metrics[2]
                        print("best_f1：{}".format(mirco_metrics[2]))
                        self.save_model()

    def test(self):
        test_dataset, test_callback = NerDataset(file_path=self.args.test_path,
                                                 tokenizer=self.args.tokenizer,
                                                 max_len=self.args.max_seq_len,
                                                 label_list=self.args.labels)
        collate = NerCollate(max_len=self.args.max_seq_len, label2id=self.args.label2id)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 collate_fn=collate.collate_fn)
        self.load_model()
        self.model.to(self.args.device)
        with torch.no_grad():
            s_logits, e_logits = self.eval_forward(test_loader)
            role_metric, total_count = self.get_metric(s_logits, e_logits, test_callback)
            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            print(
                '[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1],
                                                                               mirco_metrics[2]))
            print(classification_report(role_metric, self.args.labels, self.args.id2label, total_count))

    def predict(self, text):
        self.load_model()
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            tokens = [i for i in text]

            encode_dict = self.args.tokenizer.encode_plus(text=tokens,
                                    max_length=self.args.max_seq_len,
                                    padding="max_length",
                                    truncating="only_first",
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            # tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(self.args.device)
            attention_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(
                self.args.device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(self.args.device)
            output = self.model(token_ids, token_type_ids, attention_mask)
            start_logits = output["ner_output"]["ner_start_logits"]
            end_logits = output["ner_output"]["ner_end_logits"]

            pred_entities = ner_decode(start_logits, end_logits, text, self.args.id2label)
            # print(dict(pred_entities))
            return dict(pred_entities)


if __name__ == '__main__':
    args = NerArgs()
    model = UIEModel(args)
    ner_pipeline = NerPipeline(model, args)

    # ner_pipeline.train()
    ner_pipeline.test()

    raw_text = "顾建国先生：研究生学历，正高级工程师，现任本公司董事长、马钢(集团)控股有限公司总经理。"
    print(raw_text)
    print(ner_pipeline.predict(raw_text))

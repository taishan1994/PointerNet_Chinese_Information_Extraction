import sys

sys.path.append('..')
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report as cr
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from UIE.model import UIEModel
from UIE.config import EeArgs
from UIE.ee_data_loader import EeDataset, EeCollate
from UIE.utils.decode import ner_decode, ner_decode2, bj_decode, sigmoid
from UIE.utils.metrics import calculate_metric, classification_report, get_p_r_f


class EePipeline:
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


    def bj_eval_forward(self, 
              data_loader, 
              label,
              id2label,
              return_report=False):
        """主体或客体"""
        s_logits, e_logits = None, None
        masks = None
        start_labels = None
        end_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(self.args.device)
            if "sbj" in self.args.tasks:
              output = self.model(re_sbj_input_ids=batch_data["re_sbj_input_ids"],
                          re_sbj_token_type_ids=batch_data["re_sbj_token_type_ids"],
                          re_sbj_attention_mask=batch_data["re_sbj_attention_mask"],
                          re_sbj_start_labels=batch_data["re_sbj_start_labels"],
                          re_sbj_end_labels=batch_data["re_sbj_end_labels"],
                          )
              start_logits = output["re_output"]["sbj_start_logits"].detach().cpu()
              end_logits = output["re_output"]["sbj_end_logits"].detach().cpu()

              tmp_mask = batch_data["re_sbj_attention_mask"].detach().cpu()
              tmp_start_labels = batch_data["re_sbj_start_labels"].detach().cpu()
              tmp_end_labels = batch_data["re_sbj_end_labels"].detach().cpu()
            else:
              output = self.model(re_obj_input_ids=batch_data["re_obj_input_ids"],
                          re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                          re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                          re_obj_start_labels=batch_data["re_obj_start_labels"],
                          re_obj_end_labels=batch_data["re_obj_end_labels"],
                          )
              start_logits = output["re_output"]["obj_start_logits"].detach().cpu()
              end_logits = output["re_output"]["obj_end_logits"].detach().cpu()

              tmp_mask = batch_data["re_obj_attention_mask"].detach().cpu()
              tmp_start_labels = batch_data["re_obj_start_labels"].detach().cpu()
              tmp_end_labels = batch_data["re_obj_end_labels"].detach().cpu()
            

            if start_labels is None:
                s_logits = start_logits
                e_logits = end_logits
                masks = tmp_mask
                start_labels = tmp_start_labels
                end_labels = tmp_end_labels
            else:
                s_logits = np.append(s_logits, start_logits, axis=0)
                e_logits = np.append(e_logits, end_logits, axis=0)
                masks = np.append(masks, tmp_mask, axis=0)
                start_labels = np.append(start_labels, tmp_start_labels, axis=0)
                end_labels = np.append(end_labels, tmp_end_labels, axis=0)

        bj_outputs = {
            "s_logits": s_logits,
            "e_logits": e_logits,
            "masks": masks,
            "start_labels": start_labels,
            "end_labels": end_labels,
        }

        metrics = {}
        

        metrics = self.get_bj_metrics(bj_outputs, 
                        label=label,
                        id2label=id2label,
                        return_report=return_report)
        metrics["bj_metrics"] = metrics

        return metrics


    def eval_forward(self, 
            data_loader, 
            label,
            id2label,
            return_report=False):
        ner_s_logits, ner_e_logits = [], []
        ner_masks = None
        ner_start_labels = None
        ner_end_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(self.args.device)
            output = self.model(ner_input_ids=batch_data["ner_input_ids"],
                        ner_token_type_ids=batch_data["ner_token_type_ids"],
                        ner_attention_mask=batch_data["ner_attention_mask"],
                        ner_start_labels=batch_data["ner_start_labels"],
                        ner_end_labels=batch_data["ner_end_labels"],
                        )
            
            ner_start_logits = output["ner_output"]["ner_start_logits"]
            ner_end_logits = output["ner_output"]["ner_end_logits"]
            batch_size = batch_data['ner_input_ids'].size(0)
            for i in range(batch_size):
                ner_s_logits.append([logit[i, :] for logit in ner_start_logits])
                ner_e_logits.append([logit[i, :] for logit in ner_end_logits])

            tmp_ner_mask = batch_data["ner_attention_mask"].detach().cpu()
            tmp_ner_start_labels = batch_data["ner_start_labels"].detach().cpu()
            tmp_ner_end_labels = batch_data["ner_end_labels"].detach().cpu()

            if ner_start_labels is None:

                ner_masks = tmp_ner_mask
                ner_start_labels = tmp_ner_start_labels
                ner_end_labels = tmp_ner_end_labels
            else:

                ner_masks = np.append(ner_masks, tmp_ner_mask, axis=0)
                ner_start_labels = np.append(ner_start_labels, tmp_ner_start_labels, axis=0)
                ner_end_labels = np.append(ner_end_labels, tmp_ner_end_labels, axis=0)

        ner_outputs = {
            "ner_s_logits": ner_s_logits,
            "ner_e_logits": ner_e_logits,
            "ner_masks": ner_masks,
            "ner_start_labels": ner_start_labels,
            "ner_end_labels": ner_end_labels,
        }

        metrics = {}
        
        ner_metrics = self.get_ner_metrics(ner_outputs,
                          label,
                          id2label, 
                          return_report=return_report)

        metrics["ner_metrics"] = ner_metrics

        return metrics

    def get_bj_metrics(self, 
              bj_outputs, 
              label,
              id2label,
              return_report=False):
        role_metric, total_count = self.get_bj_metrics_helper(bj_outputs, id2label)
        mirco_metrics = np.sum(role_metric, axis=0)
        mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
        res = {
          "precision": mirco_metrics[0],
          "recall": mirco_metrics[1],
          "f1": mirco_metrics[2],
          "report": None,
        }
        if return_report:
            report = classification_report(role_metric, label, id2label, total_count)
            res["report"] = report
        return res

    def get_bj_metrics_helper(self, outputs, id2label):
        total_count = [0 for _ in range(len(id2label))]
        role_metric = np.zeros([len(id2label), 3])
        s_logits = outputs["s_logits"]
        e_logits = outputs["e_logits"]
        s_label = outputs["start_labels"]
        e_label = outputs["end_labels"]
        masks = outputs["masks"]
        for s_logit, e_logit, s_label, e_label, mask in zip(s_logits, e_logits, s_label, e_label, masks):
            length = sum(mask)
            pred_entities = bj_decode(s_logit, e_logit, length, id2label)
            true_entities = bj_decode(s_label, e_label, length, id2label)
            # print("========================")
            # print(pred_entities)
            # print(true_entities)
            # print("========================")
            for idx, _type in enumerate(list(id2label.values())):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                role_metric[idx] += calculate_metric(pred_entities[_type], true_entities[_type])
                                                     
        return role_metric, total_count

    def get_ner_metrics(self, 
              ner_outputs, 
              label,
              id2label,
              return_report=False):
        role_metric, total_count = self.get_ner_metrics_helper(ner_outputs)
        mirco_metrics = np.sum(role_metric, axis=0)
        mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
        res = {
          "precision": mirco_metrics[0],
          "recall": mirco_metrics[1],
          "f1": mirco_metrics[2],
          "report": None,
        }
        if return_report:
            report = classification_report(role_metric, label, id2label, total_count)
            res["report"] = report
        return res

    def get_ner_metrics_helper(self, ner_outputs):
        total_count = [0 for _ in range(len(self.args.ent_id2label))]
        role_metric = np.zeros([len(self.args.ent_id2label), 3])
        s_logits = ner_outputs["ner_s_logits"]
        e_logits = ner_outputs["ner_e_logits"]
        s_label = ner_outputs["ner_start_labels"]
        e_label = ner_outputs["ner_end_labels"]
        masks = ner_outputs["ner_masks"]
        for s_logit, e_logit, s_label, e_label, mask in zip(s_logits, e_logits, s_label, e_label, masks):
            length = sum(mask)
            pred_entities = ner_decode2(s_logit, e_logit, length, self.args.ent_id2label)
            true_entities = ner_decode2(s_label, e_label, length, self.args.ent_id2label)
            # print("========================")
            # print(pred_entities)
            # print(true_entities)
            # print("========================")
            for idx, _type in enumerate(self.args.entity_label):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                role_metric[idx] += calculate_metric(pred_entities[_type], true_entities[_type])
                                                     
        return role_metric, total_count

    def train(self, dev=True):
        train_dataset = EeDataset(file_path=self.args.train_path,
                      tokenizer=self.args.tokenizer,
                      max_len=self.args.max_seq_len,
                      entity_label=self.args.entity_label,
                      tasks=self.args.tasks)
        collate = EeCollate(max_len=self.args.max_seq_len, tokenizer=self.args.tokenizer, tasks=self.args.tasks)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  sampler=train_sampler,
                                  num_workers=2,
                                  collate_fn=collate.collate_fn)
        dev_loader = None
        dev_callback = None
        if dev:
            dev_dataset = EeDataset(file_path=self.args.dev_path,
                        tokenizer=self.args.tokenizer,
                        max_len=self.args.max_seq_len,
                        entity_label=self.args.entity_label,
                        tasks=self.args.tasks)
            dev_dataset = dev_dataset
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
                if "ner" in self.args.tasks:
                  output = self.model(ner_input_ids=batch_data["ner_input_ids"],
                            ner_token_type_ids=batch_data["ner_token_type_ids"],
                            ner_attention_mask=batch_data["ner_attention_mask"],
                            ner_start_labels=batch_data["ner_start_labels"],
                            ner_end_labels=batch_data["ner_end_labels"],
                            )
                  loss = output["ner_output"]["ner_loss"]
                elif "obj" in self.args.tasks:
                  output = self.model(
                            re_obj_input_ids=batch_data["re_obj_input_ids"],
                            re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                            re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                            re_obj_start_labels=batch_data["re_obj_start_labels"],
                            re_obj_end_labels=batch_data["re_obj_end_labels"],
                            )
                  loss = output["re_output"]["obj_loss"]
                  
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print('【train】 Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))
                if dev and global_step % eval_step == 0:
                  if "ner" in self.args.tasks:
                    metrics = self.eval_forward(dev_loader, self.args.entity_label, self.args.ent_id2label)
                    ner_metrics = metrics["ner_metrics"]
                    print('【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(ner_metrics["precision"],
                                                          ner_metrics["recall"],
                                                          ner_metrics["f1"]))
                    if ner_metrics["f1"] > best_f1:
                        best_f1 = ner_metrics["f1"]
                        print("【best_f1】：{}".format(best_f1))
                        self.save_model()

                  elif "obj" in self.args.tasks:
                    label = ["答案"]
                    id2label = {0:"答案"}
                    metrics = self.bj_eval_forward(dev_loader, label, id2label)
                    bj_metrics = metrics["bj_metrics"]
                    print('【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(bj_metrics["precision"],
                                                          bj_metrics["recall"],
                                                          bj_metrics["f1"]))
                    if bj_metrics["f1"] > best_f1:
                        best_f1 = bj_metrics["f1"]
                        print("【best_f1】：{}".format(best_f1))
                        self.save_model()

                  


    def test(self):
        test_dataset = EeDataset(file_path=self.args.test_path,
                      tokenizer=self.args.tokenizer,
                      max_len=self.args.max_seq_len,
                      entity_label=self.args.entity_label,
                      tasks=self.args.tasks)
        collate = EeCollate(max_len=self.args.max_seq_len, tokenizer=self.args.tokenizer, tasks=self.args.tasks)
        test_dataset = test_dataset
        test_loader = DataLoader(dataset=test_dataset,
                          batch_size=self.args.eval_batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=collate.collate_fn)
        self.load_model()
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
          if "ner" in self.args.tasks:
            metrics = self.eval_forward(test_loader, self.args.entity_label, self.args.ent_id2label, return_report=True)
            ner_metrics = metrics["ner_metrics"]
            print('【test】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(ner_metrics["precision"],
                                                            ner_metrics["recall"],
                                                            ner_metrics["f1"]))
            print(ner_metrics["report"])
          elif "obj" in self.args.tasks:
            label = ["答案"]
            id2label = {0:"答案"}
            metrics = self.bj_eval_forward(test_loader, label, id2label, return_report=True)
            bj_metrics = metrics["bj_metrics"]
            print('【test】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(bj_metrics["precision"],
                                                            bj_metrics["recall"],
                                                            bj_metrics["f1"]))
            print(bj_metrics["report"])



    def predict(self, textb, texta=None):
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            if "ner" in self.args.tasks:
              tokens_b = [i for i in textb]
              encode_dict = self.args.tokenizer.encode_plus(text=tokens_b,
                                      max_length=self.args.max_seq_len,
                                      padding="max_length",
                                      truncation="only_first",
                                      return_token_type_ids=True,
                                      return_attention_mask=True)
              # tokens = ['[CLS]'] + tokens + ['[SEP]']
              token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(self.args.device)
              attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(
                  self.args.device)
              token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(self.args.device)
              output = self.model(
                ner_input_ids=token_ids,
                ner_token_type_ids=token_type_ids,
                ner_attention_mask=attention_masks,
              )
              start_logits = output["ner_output"]["ner_start_logits"]
              end_logits = output["ner_output"]["ner_end_logits"]
              pred_entities = ner_decode(start_logits, end_logits, textb, self.args.ent_id2label)
              return dict(pred_entities)

            elif "obj" in self.args.tasks:
              tokens = [i for i in texta] + ['[SEP]'] + [i for i in textb]
              ori_tokens = tokens
              attention_mask = [1] * len(tokens)
              token_type_ids = [0] * self.args.max_seq_len
              if len(tokens) > self.args.max_seq_len - 2:
                tokens = tokens[:self.args.max_seq_len-2]
                attention_mask = attention_mask[:self.args.max_seq_len-2]
              tokens = ['[CLS]'] + tokens + ['[SEP]']
              token_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
              token_ids = token_ids + [0] * (self.args.max_seq_len - len(token_ids))
              attention_mask = attention_mask + [0] * (self.args.max_seq_len - len(attention_mask))
              token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.args.device)
              attention_mask = torch.from_numpy(np.array(attention_mask)).unsqueeze(0).to(
                  self.args.device)
              token_type_ids = torch.from_numpy(np.array(token_type_ids)).unsqueeze(0).to(self.args.device)

              output = self.model(
                re_obj_input_ids=token_ids,
                re_obj_token_type_ids=token_type_ids,
                re_obj_attention_mask=attention_mask,
              )
              start_logits = output["re_output"]["obj_start_logits"].detach().cpu()
              end_logits = output["re_output"]["obj_end_logits"].detach().cpu()
              length = sum(attention_mask.detach().cpu()[0])


              pred_entities = bj_decode(start_logits, end_logits, length, {0:"答案"})
              values = pred_entities["答案"]
              objects = []
              for v in values:
                start = v[0]
                end = v[1]
                objects.append("".join(ori_tokens[start:end]))
              return objects
              



if __name__ == '__main__':
    args = EeArgs()
    model = UIEModel(args)
    ee_pipeline = EePipeline(model, args)

    # ee_pipeline.train()
    ee_pipeline.test()

    ee_pipeline.load_model()
    if "ner" in args.tasks:
      raw_text = "富国银行收缩农业与能源贷款团队 裁减200多名银行家"
      print(raw_text)
      print(ee_pipeline.predict(raw_text))
    elif "obj" in args.tasks:
      textb = "富国银行收缩农业与能源贷款团队 裁减200多名银行家"
      texta = "组织关系-裁员_裁员人数"
      texta = "组织关系-裁员_裁员方"
      print(textb)
      print(texta)
      print(ee_pipeline.predict(textb, texta))

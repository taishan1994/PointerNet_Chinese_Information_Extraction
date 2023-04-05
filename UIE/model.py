import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class UIEModel(nn.Module):
    def __init__(self, args):
        super(UIEModel, self).__init__()
        self.args = args
        
        self.tasks = args.tasks
        bert_dir = args.bert_dir
        self.bert_config = BertConfig.from_pretrained(bert_dir)
        self.bert_model = BertModel.from_pretrained(bert_dir)

        if "ner" in args.tasks:
            self.ner_num_labels = args.ner_num_labels
            self.module_start_list = nn.ModuleList()
            self.module_end_list = nn.ModuleList()
            for i in range(args.ner_num_labels):
                self.module_start_list.append(nn.Linear(self.bert_config.hidden_size, 1))
                self.module_end_list.append(nn.Linear(self.bert_config.hidden_size, 1))
            self.ner_criterion = nn.BCEWithLogitsLoss()

        if "sbj" in args.tasks:
            self.re_sbj_start_fc = nn.Linear(self.bert_config.hidden_size, 1)
            self.re_sbj_end_fc = nn.Linear(self.bert_config.hidden_size, 1)
        if "obj" in args.tasks:
            self.re_obj_start_fc = nn.Linear(self.bert_config.hidden_size, 1)
            self.re_obj_end_fc = nn.Linear(self.bert_config.hidden_size, 1)
        if "rel" in args.tasks:
            self.re_num_labels = args.re_num_labels
            self.re_rel_fc = nn.Linear(self.bert_config.hidden_size, self.re_num_labels)
            
        self.rel_criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['ner_input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(32, 56)))
        inputs['ner_attention_mask'] = torch.ones(size=(32, 56)).long()
        inputs['ner_token_type_ids'] = torch.zeros(size=(32, 56)).long()
        inputs['ner_start_labels'] = torch.zeros(size=(32, 8, 56)).float()
        inputs['ner_end_labels'] = torch.zeros(size=(32, 8, 56)).float()

        inputs['re_sbj_input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(32, 56)))
        inputs['re_sbj_attention_mask'] = torch.ones(size=(32, 56)).long()
        inputs['re_sbj_token_type_ids'] = torch.zeros(size=(32, 56)).long()
        inputs['re_sbj_start_labels'] = torch.zeros(size=(32, 56)).float()
        inputs['re_sbj_end_labels'] = torch.zeros(size=(32, 56)).float()

        inputs['re_obj_input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(32, 56)))
        inputs['re_obj_attention_mask'] = torch.ones(size=(32, 56)).long()
        inputs['re_obj_token_type_ids'] = torch.zeros(size=(32, 56)).long()
        inputs['re_obj_start_labels'] = torch.zeros(size=(32, 56)).float()
        inputs['re_obj_end_labels'] = torch.zeros(size=(32, 56)).float()

        inputs['re_rel_input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(32, 56)))
        inputs['re_rel_attention_mask'] = torch.ones(size=(32, 56)).long()
        inputs['re_rel_token_type_ids'] = torch.zeros(size=(32, 56)).long()
        inputs['re_rel_labels'] = torch.zeros(size=(32, 16)).float()
        return inputs

    def get_pointer_loss(self,
                         start_logits,
                         end_logits,
                         attention_mask,
                         start_labels,
                         end_labels,
                         criterion):

        start_logits = start_logits.view(-1)
        end_logits = end_logits.view(-1)

        active_loss = attention_mask.view(-1) == 1

        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]

        active_start_labels = start_labels.view(-1)[active_loss]
        active_end_labels = end_labels.view(-1)[active_loss]

        start_loss = criterion(active_start_logits, active_start_labels)
        end_loss = criterion(active_end_logits, active_end_labels)
        loss = start_loss + end_loss

        return loss

    def re_obj_forward(self,
                       re_obj_input_ids=None,
                       re_obj_token_type_ids=None,
                       re_obj_attention_mask=None,
                       re_obj_start_labels=None,
                       re_obj_end_labels=None, ):
        res = {
            "obj_start_logits": None,
            "obj_end_logits": None,
            "obj_loss": None,
        }
        obj_output = self.bert_model(
            re_obj_input_ids,
            re_obj_token_type_ids,
            re_obj_attention_mask,
        )
        obj_output = obj_output[0]
        obj_start_logits = self.re_obj_start_fc(obj_output).squeeze()
        obj_end_logits = self.re_obj_end_fc(obj_output).squeeze()
        res["obj_start_logits"] = obj_start_logits
        res["obj_end_logits"] = obj_end_logits
        if re_obj_start_labels is not None and re_obj_end_labels is not None:
            obj_loss = self.get_pointer_loss(
                obj_start_logits,
                obj_end_logits,
                re_obj_attention_mask,
                re_obj_start_labels,
                re_obj_end_labels,
                self.rel_criterion,
            )

            res["obj_loss"] = obj_loss
        return res

    def re_rel_forward(self,
                       re_rel_input_ids=None,
                       re_rel_token_type_ids=None,
                       re_rel_attention_mask=None,
                       re_rel_labels=None
                       ):
        res = {
            "rel_logits": None,
            "rel_loss": None,
        }
        rel_output = self.bert_model(
            re_rel_input_ids,
            re_rel_token_type_ids,
            re_rel_attention_mask,
        )

        # 选择句子级别的输出
        rel_output = rel_output[1]
        re_rel_logits = self.re_rel_fc(rel_output)
        res["rel_logits"] = re_rel_logits
        if re_rel_labels is not None:
            rel_loss = self.rel_criterion(re_rel_logits, re_rel_labels)
            res["rel_loss"] = rel_loss

        return res

    def re_sbj_forward(self,
                       re_sbj_input_ids=None,
                       re_sbj_token_type_ids=None,
                       re_sbj_attention_mask=None,
                       re_sbj_start_labels=None,
                       re_sbj_end_labels=None,
                       ):
        res = {
            "sbj_start_logits": None,
            "sbj_end_logits": None,
            "sbj_loss": None
        }
        sbj_output = self.bert_model(
            re_sbj_input_ids,
            re_sbj_token_type_ids,
            re_sbj_attention_mask,
        )
        sbj_output = sbj_output[0]
        sbj_start_logits = self.re_sbj_start_fc(sbj_output).squeeze()
        sbj_end_logits = self.re_sbj_end_fc(sbj_output).squeeze()
        res["sbj_start_logits"] = sbj_start_logits
        res["sbj_end_logits"] = sbj_end_logits

        if re_sbj_start_labels is not None and re_sbj_end_labels is not None:
            sbj_loss = self.get_pointer_loss(
                sbj_start_logits,
                sbj_end_logits,
                re_sbj_attention_mask,
                re_sbj_start_labels,
                re_sbj_end_labels,
                self.rel_criterion,
            )

            res["sbj_loss"] = sbj_loss

        return res

    def ner_forward(self,
                    ner_input_ids,
                    ner_token_type_ids,
                    ner_attention_mask,
                    ner_start_labels=None,
                    ner_end_labels=None):

        bert_output = self.bert_model(
            input_ids=ner_input_ids,
            token_type_ids=ner_token_type_ids,
            attention_mask=ner_attention_mask)
        seq_bert_output = bert_output[0]

        all_start_logits = []
        all_end_logits = []
        ner_loss = None

        res = {
            "ner_start_logits": None,
            "ner_end_logits": None,
            "ner_loss": None
        }

        for i in range(self.ner_num_labels):
            start_logits = self.module_start_list[i](seq_bert_output).squeeze()
            end_logits = self.module_end_list[i](seq_bert_output).squeeze()

            all_start_logits.append(start_logits.detach().cpu())
            all_end_logits.append(end_logits.detach().cpu())

            if ner_start_labels is not None and ner_start_labels is not None:
                start_logits = start_logits.view(-1)
                end_logits = end_logits.view(-1)

                active_loss = ner_attention_mask.view(-1) == 1

                active_start_logits = start_logits[active_loss]
                active_end_logits = end_logits[active_loss]

                active_start_labels = ner_start_labels[:, i, :].contiguous().view(-1)[active_loss]
                active_end_labels = ner_end_labels[:, i, :].contiguous().view(-1)[active_loss]

                start_loss = self.ner_criterion(active_start_logits, active_start_labels)
                end_loss = self.ner_criterion(active_end_logits, active_end_labels)
                if ner_loss is None:
                    ner_loss = start_loss + end_loss
                else:
                    ner_loss += (start_loss + end_loss)

        res["ner_start_logits"] = all_start_logits
        res["ner_end_logits"] = all_end_logits
        res["ner_loss"] = ner_loss

        return res

    def forward(self,
                ner_input_ids=None,
                ner_token_type_ids=None,
                ner_attention_mask=None,
                ner_start_labels=None,
                ner_end_labels=None,
                re_sbj_input_ids=None,
                re_sbj_token_type_ids=None,
                re_sbj_attention_mask=None,
                re_obj_input_ids=None,
                re_obj_token_type_ids=None,
                re_obj_attention_mask=None,
                re_rel_input_ids=None,
                re_rel_token_type_ids=None,
                re_rel_attention_mask=None,
                re_sbj_start_labels=None,
                re_sbj_end_labels=None,
                re_obj_start_labels=None,
                re_obj_end_labels=None,
                re_rel_labels=None,
                ee_start_labels=None,
                ee_end_labels=None):

        res = {
            "ner_output": None,
            "re_output": None,
            "event_output": None
        }

        if "ner" in self.tasks:
            ner_output = self.ner_forward(
                ner_input_ids,
                ner_token_type_ids,
                ner_attention_mask,
                ner_start_labels,
                ner_end_labels,
            )
            res["ner_output"] = ner_output

        elif "sbj" in self.tasks:
            re_output = self.re_sbj_forward(
                re_sbj_input_ids=re_sbj_input_ids,
                re_sbj_token_type_ids=re_sbj_token_type_ids,
                re_sbj_attention_mask=re_sbj_attention_mask,
                re_sbj_start_labels=re_sbj_start_labels,
                re_sbj_end_labels=re_sbj_end_labels,
            )
            res["re_output"] = re_output

        elif "obj" in self.tasks:
            re_output = self.re_obj_forward(
                re_obj_input_ids=re_obj_input_ids,
                re_obj_token_type_ids=re_obj_token_type_ids,
                re_obj_attention_mask=re_obj_attention_mask,
                re_obj_start_labels=re_obj_start_labels,
                re_obj_end_labels=re_obj_end_labels,
            )
            res["re_output"] = re_output

        elif "rel" in self.tasks:
            re_output = self.re_rel_forward(
                re_rel_input_ids=re_rel_input_ids,
                re_rel_token_type_ids=re_rel_token_type_ids,
                re_rel_attention_mask=re_rel_attention_mask,
                re_rel_labels=re_rel_labels,
            )
            res["re_output"] = re_output

        return res


if __name__ == '__main__':
    inputs = UIEModel.build_dummpy_inputs()


    class Args:
        bert_dir = "../chinese-bert-wwm-ext/"
        ner_num_labels = 8
        re_num_labels = 16
        tasks = ["re_rel"]


    args = Args()
    model = UIEModel(args)
    res = model(
        ner_input_ids=inputs["ner_input_ids"],
        ner_token_type_ids=inputs["ner_token_type_ids"],
        ner_attention_mask=inputs["ner_attention_mask"],
        ner_start_labels=inputs["ner_start_labels"],
        ner_end_labels=inputs["ner_end_labels"],
        re_sbj_input_ids=inputs["re_sbj_input_ids"],
        re_sbj_token_type_ids=inputs["re_sbj_token_type_ids"],
        re_sbj_attention_mask=inputs["re_sbj_attention_mask"],
        re_sbj_start_labels=inputs["re_sbj_start_labels"],
        re_sbj_end_labels=inputs["re_sbj_end_labels"],
        re_obj_input_ids=inputs["re_obj_input_ids"],
        re_obj_token_type_ids=inputs["re_obj_token_type_ids"],
        re_obj_attention_mask=inputs["re_obj_attention_mask"],
        re_obj_start_labels=inputs["re_obj_start_labels"],
        re_obj_end_labels=inputs["re_obj_end_labels"],
        re_rel_input_ids=inputs["re_rel_input_ids"],
        re_rel_token_type_ids=inputs["re_rel_token_type_ids"],
        re_rel_attention_mask=inputs["re_rel_attention_mask"],
        re_rel_labels=inputs["re_rel_labels"],
    )

    print(res)

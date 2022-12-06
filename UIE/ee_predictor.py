import sys
sys.path.append("..")
import json
import torch
from transformers import BertTokenizer
from UIE.ee_main import EePipeline
from UIE.model import UIEModel


class CommonArgs:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  bert_dir = "model_hub/chinese-bert-wwm-ext/"
  tokenizer = BertTokenizer.from_pretrained(bert_dir)
  max_seq_len = 256
  data_name = "duee"



class NerArgs:
  tasks = ["ner"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/ee/{}_{}_model.pt".format(tasks[0], data_name)
  entity_label_path = "./data/ee/{}/labels.txt".format(data_name)
  with open(entity_label_path, "r", encoding="utf-8") as fp:
      entity_label = fp.read().strip().split("\n")
  ner_num_labels = len(entity_label)
  ent_label2id = {}
  ent_id2label = {}
  for i, label in enumerate(entity_label):
      ent_label2id[label] = i
      ent_id2label[i] = label



class ObjArgs:
  tasks = ["obj"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/ee/{}_{}_model.pt".format(tasks[0], data_name)
  label2role_path = "./data/ee/{}/label2role.json".format(data_name)
  with open(label2role_path, "r", encoding="utf-8") as fp:
    label2role = json.load(fp)


  

ner_args = NerArgs()
obj_args = ObjArgs()


class Predictor:
  def __init__(self, ner_args=None, obj_args=None):
    model = UIEModel(ner_args)
    self.ner_pipeline = EePipeline(model, ner_args)
    self.ner_pipeline.load_model()

    model = UIEModel(obj_args)
    self.obj_pipeline = EePipeline(model, obj_args)
    self.obj_pipeline.load_model()

  def predict_ner(self, text):
    entities = self.ner_pipeline.predict(text)
    return entities

  def predict_obj(self, text, subjects):
    sbj_obj = []
    for sbj in subjects:
      objects = self.obj_pipeline.predict(text, sbj)
      for obj in objects:
        sbj_obj.append([sbj, obj])
    return sbj_obj




if __name__ == "__main__":
  text = "青岛地铁施工段坍塌致3人遇难 2名失联者仍在搜救"
  text = "2019年7月12日，国家市场监督管理总局缺陷产品管理中心，在其官方网站和微信公众号上发布了《上海施耐德低压终端电器有限公司召回部分剩余电流保护装置》，看到这条消息，确实令人震惊！\n作为传统的三大外资品牌之一，竟然发生如此大规模质量问题的召回，而且生产持续时间长达一年！从采购，检验，生产，测试，包装，销售，这么多环节竟没有反馈出问题，处于无人知晓状态，问题出在哪里？希望官方能有一个解释了。"
  predict_tool = Predictor(ner_args, obj_args)

  print("文本：", text)
  entities = predict_tool.predict_ner(text)
  event_types = []
  print("实体：")
  for k,v in entities.items():
    if len(v) != 0:
      print(k, v)
      event_types.append(k)
  for event_type in event_types:
    print("事件类型：", event_type)
    subjects = obj_args.label2role[event_type]
    sbj_obj = predict_tool.predict_obj(text, subjects)
    print("实体：", sbj_obj)
    print("="*100)





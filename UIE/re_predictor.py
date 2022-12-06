import sys
sys.path.append("..")
import torch
from transformers import BertTokenizer
from UIE.re_main import RePipeline
from UIE.model import UIEModel


class CommonArgs:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  bert_dir = "model_hub/chinese-bert-wwm-ext/"
  tokenizer = BertTokenizer.from_pretrained(bert_dir)
  max_seq_len = 256
  data_name = "ske"



class NerArgs:
  tasks = ["ner"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)
  entity_label_path = "./data/re/{}/entity_labels.txt".format(data_name)
  with open(entity_label_path, "r", encoding="utf-8") as fp:
      entity_label = fp.read().strip().split("\n")
  ner_num_labels = len(entity_label)
  ent_label2id = {}
  ent_id2label = {}
  for i, label in enumerate(entity_label):
      ent_label2id[label] = i
      ent_id2label[i] = label


class SbjArgs:
  tasks = ["sbj"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)


class ObjArgs:
  tasks = ["obj"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)


class RelArgs:
  tasks = ["rel"]
  device = CommonArgs.device
  bert_dir = CommonArgs.bert_dir
  tokenizer = CommonArgs.tokenizer
  max_seq_len = CommonArgs.max_seq_len
  data_name = CommonArgs.data_name
  save_dir = "./checkpoints/re/{}_{}_model.pt".format(tasks[0], data_name)
  relation_label_path = "./data/re/{}/relation_labels.txt".format(data_name)
  with open(relation_label_path, "r", encoding='utf-8') as fp:
        relation_label = fp.read().strip().split("\n")
  relation_label.append("没有关系")
  rel_label2id = {}
  rel_id2label = {}
  for i, label in enumerate(relation_label):
      rel_label2id[label] = i
      rel_id2label[i] = label

  re_num_labels = len(relation_label)
  

ner_args = NerArgs()
sbj_args = SbjArgs()
obj_args = ObjArgs()
rel_args = RelArgs()


class Predictor:
  def __init__(self, ner_args=None, sbj_args=None, obj_args=None, rel_args=None):
    model = UIEModel(ner_args)
    self.ner_pipeline = RePipeline(model, ner_args)
    self.ner_pipeline.load_model()

    model = UIEModel(sbj_args)
    self.sbj_pipeline = RePipeline(model, sbj_args)
    self.sbj_pipeline.load_model()

    model = UIEModel(obj_args)
    self.obj_pipeline = RePipeline(model, obj_args)
    self.obj_pipeline.load_model()

    model = UIEModel(rel_args)
    self.rel_pipeline = RePipeline(model, rel_args)
    self.rel_pipeline.load_model()

  def predict_ner(self, text):
    entities = self.ner_pipeline.predict(text)
    return entities

  def predict_sbj(self, text):
    subjects = self.sbj_pipeline.predict(text)
    return subjects

  def predict_obj(self, text, subjects):
    sbj_obj = []
    for sbj in subjects:
      objects = self.obj_pipeline.predict(text, sbj)
      for obj in objects:
        sbj_obj.append([sbj, obj])
    return sbj_obj

  def predict_rel(self, text, sbj_obj):
    sbj_rel_obj = []
    for so in sbj_obj:
      rels = self.rel_pipeline.predict(text, "#;#".join(so))
      for rel in rels:
        sbj_rel_obj.append((so[0], rel, so[1]))
    return sbj_rel_obj




if __name__ == "__main__":
  text = "《邪少王兵》是冰火未央写的网络小说连载于旗峰天下"
  text = "《父老乡亲》是由是由由中国人民解放军海政文工团创作的军旅歌曲，石顺义作词，王锡仁作曲，范琳琳演唱"
  text = "《外国民间歌曲选》是2004年人民音乐出版社出版的图书，作者是温恒泰"
  text = "王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人"
  text = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"
  text = "马志舟，1907年出生，陕西三原人，汉族，中国共产党，任红四团第一连连长，1933年逝世"
  text = "《你的嘴》收录于歌手金莎的音乐专辑《星月神话》，由许嵩作词作曲，2010年10月15日首发"
  text = "红地球葡萄栽培技术问答是一本由刘洪章主编，天津科技翻译出版公司出版的关于红地球葡萄标准化、产业化栽培技术为核心的图书"
  text = "《神之水滴》改编自亚树直的同名漫画，是日本电视台2009年1月13日制作并播放的电视剧，共九集"
  predict_tool = Predictor(ner_args, sbj_args, obj_args, rel_args)

  print("文本：", text)
  entities = predict_tool.predict_ner(text)
  print("实体：")
  for k,v in entities.items():
    if len(v) != 0:
      print(k, v)
  subjects = predict_tool.predict_sbj(text)
  print("主体：", subjects)
  sbj_obj = predict_tool.predict_obj(text, subjects)
  print("客体：", sbj_obj)
  sbj_rel_obj = predict_tool.predict_rel(text, sbj_obj)
  print("关系：", sbj_rel_obj)




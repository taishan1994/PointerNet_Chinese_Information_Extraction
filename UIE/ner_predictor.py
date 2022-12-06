import sys
sys.path.append("..")
import torch
from transformers import BertTokenizer
from UIE.ner_main import NerPipeline
from UIE.model import UIEModel


class NerArgs:
    tasks = ["ner"]
    data_name = "cner"
    data_dir = "ner"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model.pt".format(data_dir, tasks[0], data_name)
    label_path = "./data/{}/{}/labels.txt".format(data_dir, data_name)
    with open(label_path, "r") as fp:
        labels = fp.read().strip().split("\n")
    label2id = {}
    id2label = {}
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    ner_num_labels = len(labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    max_seq_len = 150

class Predictor:
  def __init__(self, ner_args=None):
    model = UIEModel(ner_args)
    self.ner_pipeline = NerPipeline(model, ner_args)
    self.ner_pipeline.load_model()

  def predict_ner(self, text):
    entities = self.ner_pipeline.predict(text)
    return entities


nerArgs = NerArgs()
predict_tool = Predictor(nerArgs)
text = "顾建国先生：研究生学历，正高级工程师，现任本公司董事长、马钢(集团)控股有限公司总经理。"
print("文本：", text)
entities = predict_tool.predict_ner(text)
print("实体：")
for k,v in entities.items():
  if len(v) != 0:
    print(k, v)



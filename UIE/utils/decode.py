import numpy as np
from collections import defaultdict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ner_decode(start_logits, end_logits, raw_text, id2label):
    predict_entities = defaultdict(list)
    # print(start_pred)
    # print(end_pred)
    for label_id in range(len(id2label)):
      start_logit = np.where(start_logits[label_id] > 0.5, 1, 0)
      end_logit = np.where(end_logits[label_id] > 0.5, 1, 0)
      start_pred = start_logit[1:len(raw_text)+1]
      end_pred = end_logit[1:len(raw_text)+1]
      # print(raw_text)
      # print(start_pred)
      # print(end_pred)
      for i, s_type in enumerate(start_pred):
          if s_type == 0:
              continue
          for j, e_type in enumerate(end_pred[i:]):
              if s_type == e_type:
                  tmp_ent = raw_text[i:i + j + 1]
                  if tmp_ent == '':
                      continue
                  predict_entities[id2label[label_id]].append((tmp_ent, i))
                  break

    return predict_entities


def ner_decode2(start_logits, end_logits, length, id2label):
    predict_entities = {x:[] for x in list(id2label.values())}
    # predict_entities = defaultdict(list)
    # print(start_pred)
    # print(end_pred)

    for label_id in range(len(id2label)):
        start_logit = np.where(sigmoid(start_logits[label_id]) > 0.5, 1, 0)
        end_logit = np.where(sigmoid(end_logits[label_id]) > 0.5, 1, 0)
        # print(start_logit)
        # print(end_logit)
        # print("="*100)
        start_pred = start_logit[1:length + 1]
        end_pred = end_logit[1:length+ 1]
       
        for i, s_type in enumerate(start_pred):
            if s_type == 0:
                continue
            for j, e_type in enumerate(end_pred[i:]):
                if s_type == e_type:
                    predict_entities[id2label[label_id]].append((i, i+j+1))
                    break
    return predict_entities


def bj_decode(start_logits, end_logits, length, id2label):
    predict_entities = {x:[] for x in list(id2label.values())}
    start_logit = np.where(sigmoid(start_logits) > 0.5, 1, 0)
    end_logit = np.where(sigmoid(end_logits) > 0.5, 1, 0)
    start_pred = start_logit[1:length + 1]
    end_pred = end_logit[1:length+ 1]
    # print(start_pred)
    # print(end_pred)
    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                predict_entities[id2label[0]].append((i, i+j+1))
                break

    return predict_entities
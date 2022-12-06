import json
import random


def get_labels():
  labels = []
  label2role = {}
  with open("duee_event_schema.json","r",encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
  for d in data:
    d = json.loads(d)
    event_type = d["event_type"]
    labels.append(event_type)
    role_list = d["role_list"]
    for role in role_list:
      if event_type not in label2role:
        label2role[event_type] = []
      label2role[event_type].append(event_type + "_" + role["role"])
  with open("labels.txt", "w") as fp:
    fp.write("\n".join(labels))

  with open("label2role.json", "w") as fp:
    json.dump(label2role, fp, ensure_ascii=False)


def get_data(in_path, out_path, mode="train"):
  with open(in_path, "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
  res = []
  for d in data:
    d = json.loads(d)
    text = d["text"]
    text_id = d["id"]
    event_list = d["event_list"]
    tmp = {}
    tmp["id"] = text_id
    tmp["text"] = text
    tmp["labels"] = []
    ent_id = 0
    for event in event_list:
      event_type = event["event_type"]
      if "财经" in event_type:
        arguments = event["arguments"]
        for arg in arguments:
          argument_start_index = arg["argument_start_index"]
          role = arg["role"]
          argument = arg["argument"]
          tmp["labels"].append(["T{}".format(ent_id), event_type + "_" + role, argument_start_index, argument_start_index+len(argument), argument])
          ent_id += 1
    if len(tmp["labels"]) != 0:
      res.append(tmp)
    else:
      if mode == "train":
        prob = random.uniform(0, 1) 
        # 负样本
        if prob > 0.9:
          res.append(tmp)

  with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(res, fp, ensure_ascii=False)


    


get_labels()
# get_data("duee_train.json", "train.json", "train")
# get_data("duee_dev.json", "dev.json", "dev")


  
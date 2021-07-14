# -*- coding: utf-8 -*-
# @Time    : 2021/4/15 13:27
# @Author  : zxf
import json

import tensorflow as tf
from flask import Flask
from flask import request
from flask import jsonify
from tensorflow.python.keras.backend import set_session

import parse as args
from model import E2EModel
from utils import get_tokenizer
from utils import extract_items

app = Flask(__name__)
sess = tf.Session()
set_session(sess)
graph = tf.get_default_graph()

id2rel, rel2id = json.load(open("./data/CCKS2019/rel2id.json", "r", encoding="utf-8"))
id2rel = {int(i): j for i, j in id2rel.items()}
num_rels = len(id2rel)
tokenizer = get_tokenizer(args.bert_vocab_path)
subject_model, object_model, hbt_model = E2EModel(args.bert_config_path,
                                                      args.bert_checkpoint_path,
                                                      args.LR, num_rels)
hbt_model.load_weights(args.save_weights_path)


@app.route("/relation", methods=['POST'])
def get_casrel_model_predict():
    data = json.loads(request.get_data(), encoding="utf-8")
    sent = data.get("input")
    # 提前加载模型会报错，为什么
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        result = extract_items(subject_model, object_model, tokenizer,
                           sent, id2rel, h_bar=0.5, t_bar=0.5)
    return_result = {"errcode": 200,
                     "content": sent,
                     "relation": result}
    print("result: ", return_result)
    return jsonify(return_result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 18:52
# @Author  : zxf
import os
import json

import parse as args
import tensorflow as tf
from model import E2EModel
from utils import extract_items
from data_loader import to_tuple
from utils import get_tokenizer, metric

from keras import backend as K


def load_data(test_path, rel_dict_path):
    test_data = json.load(open(test_path, "r", encoding="utf-8"))
    id2rel, rel2id = json.load(open(rel_dict_path, "r", encoding="utf-8"))

    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)

    for sent in test_data:
        to_tuple(sent)

    return test_data, id2rel, rel2id, num_rels


def model_predict():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if K.backend() == 'tensorflow':
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    tokenizer = get_tokenizer(args.bert_vocab_path)
    # read data and relation
    test_data, id2rel, rel2id, num_rels = load_data(args.test_path, args.rel_dict_path)
    # load model
    subject_model, object_model, hbt_model = E2EModel(args.bert_config_path,
                                                      args.bert_checkpoint_path,
                                                      args.LR, num_rels)
    hbt_model.load_weights(args.save_weights_path)

    isExactMatch = True if args.dataset == 'Wiki-KBP' else False
    if isExactMatch:
        print("Exact Match")
    else:
        print("Partial Match")
    precision, recall, f1_score = metric(subject_model, object_model, test_data, id2rel, tokenizer,
                                         isExactMatch, args.test_result_path)
    print(f'{precision}\t{recall}\t{f1_score}')


def predict():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if K.backend() == 'tensorflow':
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    tokenizer = get_tokenizer(args.bert_vocab_path)
    # read data and relation
    # test_data, id2rel, rel2id, num_rels = load_data(args.test_path, args.rel_dict_path)
    test_data = json.load(open(args.test_path, "r", encoding="utf-8"))
    id2rel, rel2id = json.load(open(args.rel_dict_path, "r", encoding="utf-8"))

    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)
    # load model
    subject_model, object_model, hbt_model = E2EModel(args.bert_config_path,
                                                      args.bert_checkpoint_path,
                                                      args.LR, num_rels)
    hbt_model.load_weights(args.save_weights_path)
    return_result = []
    for line in test_data:
        sent = line["text"]
        result = extract_items(subject_model, object_model, tokenizer,
                                sent, id2rel, h_bar=0.5, t_bar=0.5)
        return_result.append({
                                "text": sent,
                                "relation": result})

    with open("./results/baidurelation2020/test_data_pred.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(return_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    model_predict()
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 10:42
# @Author  : zxf
import os


# bert model
bert_model = 'chinese_L-12_H-768_A-12'
bert_config_path = 'pretrained_bert_models/' + bert_model + '/bert_config.json'
bert_vocab_path = 'pretrained_bert_models/' + bert_model + '/vocab.txt'
bert_checkpoint_path = 'pretrained_bert_models/' + bert_model + '/bert_model.ckpt'

# model parse
dataset = "CCKS2019"
train = True
train_path = 'data/' + dataset + '/train_triples_demo.json'
dev_path = 'data/' + dataset + '/dev_triples_demo.json'
test_path = 'data/' + dataset + '/test_triples_demo.json'  # overall test
rel_dict_path = 'data/' + dataset + '/rel2id.json'
save_weights_path = 'saved_weights/' + dataset + '/best_model.weights'
save_model_path = './save_model/' + dataset + '/ccks_model.h5'
h5_pb_path = "./save_model/" + dataset + '/ccks_model.pb'
pb_path = "./save_model/" + dataset + "/1/"

test_result_path = 'results/' + dataset + '/test_result.json'

LR = 1e-5
BATCH_SIZE = 2
EPOCH = 1
MAX_LEN = 128




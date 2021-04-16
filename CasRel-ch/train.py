# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 18:24
# @Author  : zxf
import os
import json

from keras import backend as K

import parse as args
from model import E2EModel, Evaluate
from data_loader import data_generator, load_data
from utils import extract_items, get_tokenizer, metric


def main():
    tokenizer = get_tokenizer(args.bert_vocab_path)
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(args.train_path,
                                                                          args.dev_path,
                                                                          args.test_path,
                                                                          args.rel_dict_path)
    subject_model, object_model, hbt_model = E2EModel(args.bert_config_path,
                                                      args.bert_checkpoint_path,
                                                      args.LR, num_rels)
    # tensorflow
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    STEPS = len(train_data) // args.BATCH_SIZE
    data_manager = data_generator(train_data, tokenizer, rel2id, num_rels,
                                  args.MAX_LEN, args.BATCH_SIZE)
    evaluator = Evaluate(subject_model, object_model, tokenizer, id2rel,
                         dev_data, args.save_weights_path, args.save_model_path)
    hbt_model.fit_generator(data_manager.__iter__(),
                            steps_per_epoch=STEPS,
                            epochs=args.EPOCH,
                            callbacks=[evaluator]
                            )
    print("model training finish")


if __name__ == "__main__":
    main()
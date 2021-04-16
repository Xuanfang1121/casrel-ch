# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 22:34
# @Author  : zxf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

import parse as args


def pbmodel_to_tf_serving(pb_model_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.info("multi classification pb model path: {}".format(output_path))

    builder = tf.saved_model.builder.SavedModelBuilder(output_path)
    with tf.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")

        g = tf.get_default_graph()

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                inputs={"input_1": g.get_operation_by_name('input_1').outputs[0],
                        "input_2": g.get_operation_by_name('input_2').outputs[0],
                        "input_5": g.get_operation_by_name('input_5').outputs[0],
                        "input_6": g.get_operation_by_name('input_6').outputs[0]},
                outputs={"pred_sub_heads": g.get_operation_by_name('dense_1/Sigmoid').outputs[0],
                         "pred_sub_tails": g.get_operation_by_name('dense_2/Sigmoid').outputs[0],
                         "pred_obj_heads": g.get_operation_by_name('dense_3/Sigmoid').outputs[0],
                         "pred_obj_tails": g.get_operation_by_name('dense_4/Sigmoid').outputs[0]}
            )

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

        builder.save()


if __name__ == "__main__":
    pbmodel_to_tf_serving(args.h5_pb_path, args.pb_path)
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 22:28
# @Author  : zxf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import keras
import keras_metrics
from keras import backend as K
from keras.models import model_from_json
from keras_bert import get_custom_objects
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

import parse as args


def h5_to_pb_args(model_path, h5_pb_path):
    args = {'input_model': model_path,
            'input_model_json': None,
            'output_model': h5_pb_path,
            'save_graph_def': False,
            'output_nodes_prefix': None,
            'quantize': False,
            'channels_first': False,
            'output_meta_ckpt': False
    }
    return args


def load_model(input_model_path, input_json_path, custom_objects):
    if not Path(input_model_path).exists():
        raise FileNotFoundError(
            'Model file `{}` does not exist.'.format(input_model_path))
    try:
        model = keras.models.load_model(input_model_path, compile=False, custom_objects=custom_objects)
        return model
    except FileNotFoundError as err:
        logging.error('Input mode file (%s) does not exist.', input_model_path)
        raise err
    except ValueError as wrong_file_err:
        if input_json_path:
            if not Path(input_json_path).exists():
                raise FileNotFoundError(
                    'Model description json file `{}` does not exist.'.format(
                        input_json_path))
            try:
                model = model_from_json(open(str(input_json_path)).read())
                model.load_weights(input_model_path)
                return model
            except Exception as err:
                logging.error("Couldn't load model from json.")
                raise err
        else:
            logging.error(
                'Input file specified only holds the weights, and not '
                'the model definition. Save the model using '
                'model.save(filename.h5) which will contain the network '
                'architecture as well as its weights. If the model is '
                'saved using model.save_weights(filename), the flag '
                'input_model_json should also be set to the '
                'architecture which is exported separately in a '
                'json format. Check the keras documentation for more details '
                '(https://keras.io/getting-started/faq/)')
            raise wrong_file_err


def h5_to_pb(model_path, h5_pb_path):
    args = h5_to_pb_args(model_path, h5_pb_path)
    custom_objects = get_custom_objects()
    # custom_objects['binary_precision'] = keras_metrics.binary_precision()
    # custom_objects['binary_recall'] = keras_metrics.binary_recall()
    # custom_objects['binary_f1_score'] = keras_metrics.f1_score()
    K.set_learning_phase(0)
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = args["output_model"]
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent
    output_model_name = Path(output_model).name
    output_model_stem = Path(output_model).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # Create output directory if it does not exist
    # print (Path(output_model).parent)
    if not os.path.exists(str(Path(output_model).parent)):
        Path(output_model).parent.mkdir(parents=True)

    if args["channels_first"]:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

    model = load_model(args["input_model"], args["input_model_json"], custom_objects)

    input_node_names = [node.op.name for node in model.inputs]
    logging.info('Input nodes names are: %s', str(input_node_names))

    # TODO(amirabdi): Support networks with multiple inputs
    orig_output_node_names = [node.op.name for node in model.outputs]
    if args["output_nodes_prefix"]:  # 给模型节点编号
        num_output = len(orig_output_node_names)
        pred = [None] * num_output
        converted_output_node_names = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            converted_output_node_names[i] = '{}{}'.format(
                args["output_nodes_prefix"], i)
            pred[i] = tf.identity(model.outputs[i],
                                  name=converted_output_node_names[i])
    else:
        converted_output_node_names = orig_output_node_names
    logging.info('Converted output node names are: %s',
                 str(converted_output_node_names))

    sess = K.get_session()
    if args["output_meta_ckpt"]:  # 让转化的模型可以继续被训练
        saver = tf.train.Saver()
        saver.save(sess, str(output_fld / output_model_stem))

    if args["save_graph_def"]:  # 以ascii形式存储模型
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)
        logging.info('Saved the graph definition in ascii format at %s',
                     str(Path(output_fld) / output_model_pbtxt_name))

    if args["quantize"]:  # 将权重从float转为八位比特
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               converted_output_node_names,
                                               transforms)
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            transformed_graph_def,
            converted_output_node_names)
    else:  # float形式存储权重
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)
    logging.info('Saved the freezed graph at %s',
                 str(Path(output_fld) / output_model_name))


if __name__ == "__main__":
    h5_to_pb(args.save_model_path, args.h5_pb_path)
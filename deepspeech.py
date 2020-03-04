import numpy as np
import tensorflow as tf
import argparse

class DeepSpeech(object):
    """Class"""

    def __init__(self, model_path):
        print('[MD] Load model from:', model_path)
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="")

        # print out operations information
        ops = tf.get_default_graph().get_operations()
        for op in ops:
            print(op.name) # print operation name
            print('> ', op.values()) # print a list of tensors it produces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('model_path', type=str, help="Path to the DeepSpeech Model")
    # parser.add_argument('--in', type=str, dest="input",
    #                     required=True,
    #                     help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    args = parser.parse_args()

    deepspeech = DeepSpeech(args.model_path)
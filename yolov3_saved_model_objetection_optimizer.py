# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7
"""Export inception model given existing training checkpoints.
The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

from __future__ import print_function

import os.path

import tensorflow as tf
import argparse

from core.yolov3 import YOLOV3

from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python import ops

IMAGE_SIZE = 416


output_node_names = ["encoded_image_tensor", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
transforms = ['remove_nodes(op=Identity)',
              'merge_duplicate_nodes',
              'strip_unused_nodes',
              'fold_constants(ignore_errors=true)',
              'fold_batch_norms',
              'quantize_weights']  # this reduces the size, but there is no speed up , actaully slows down, see below

# transforms = ['add_default_attributes',
# 'strip_unused_nodes',
# 'remove_nodes(op=Identity, op=CheckNumerics)',
# 'fold_constants(ignore_errors=true)',
# 'fold_batch_norms',
# 'fold_old_batch_norms',
# 'quantize_weights',
# 'quantize_nodes',
# 'strip_unused_nodes',
# 'sort_by_execution_order']


parser = argparse.ArgumentParser(description='Generate a saved model.')

parser.add_argument("--model_version", default="2", help="Version number of the model.", type=str)
parser.add_argument("--output_dir", default="model/yolov3", help="export model directory", type=str)
parser.add_argument("--ckpt_dir", default="checkpoint", help="Directory where to read training checkpoints.", type=str)
parser.add_argument("--input_tensor", default="encoded_image_tensor:0", help="input tensor", type=str)
parser.add_argument("--sbbox_tensor", default="pred_sbbox/concat_2:0", help="pred_sbbox", type=str)
parser.add_argument("--mbbox_tensor", default="pred_mbbox/concat_2:0", help="pred_mbbox", type=str)
parser.add_argument("--lbbox_tensor", default="pred_lbbox/concat_2:0", help="pred_lbbox", type=str)
parser.add_argument("--class_num", default="2", help="class num", type=int)

parser.add_argument("--freeze_graph_dir", default="coco/yolov3_helmet.pb",
                    help="Directory where to save yolov3_helmet.pb.", type=str)
parser.add_argument("--optimizer_graph_dir", default="model/optimized_yolov3_helmet.pb",
                    help="Directory where to save optimized_yolov3_helmet.pb.", type=str)


args = parser.parse_args()


def get_graph_def_from_file(graph_filepath):
    tf.reset_default_graph()
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


def optimize_graph(model_dir, graph_filename, transforms, output_names, outname='optimized_model.pb'):
    input_names = ['input_image',] # change this as per how you have saved the model
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(
    graph_def,
    input_names,
    output_names,
    transforms)
    tf.train.write_graph(optimized_graph_def,
    logdir=model_dir,
    as_text=False,
    name=outname)
    print('Graph optimized!')




def export():

    serialized_tf_example = tf.placeholder(tf.string, shape=[None], name='encoded_image_tensor')

    images = tf.map_fn(preprocess_image, serialized_tf_example, tf.float32)

    model = YOLOV3(images, trainable=False)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                       input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_node_names)

    with tf.gfile.GFile(args.freeze_graph_dir, "wb") as f:
        f.write(converted_graph_def.SerializeToString())


    optimize_graph('', args.freeze_graph_dir,
                   transforms, output_node_names, outname=args.optimizer_graph_dir)

    graph_def = get_graph_def_from_file(args.optimizer_graph_dir)


    with tf.Graph().as_default():

        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
        )

        with tf.Session() as sess:

            # Export inference model.
            output_path = os.path.join(
            tf.compat.as_bytes(args.output_dir),
            tf.compat.as_bytes(str(args.model_version)))
            print('Exporting trained model to', output_path)
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            input_tensor = tf.get_default_graph().get_tensor_by_name(args.input_tensor)
            sbbox_tensor = tf.get_default_graph().get_tensor_by_name(args.sbbox_tensor)
            mbbox_tensor = tf.get_default_graph().get_tensor_by_name(args.mbbox_tensor)
            lbbox_tensor = tf.get_default_graph().get_tensor_by_name(args.lbbox_tensor)
            class_num = args.class_num

            sbbox_tensor = tf.reshape(tf.convert_to_tensor(sbbox_tensor), shape=[-1, 5 + class_num])
            mbbox_tensor = tf.reshape(tf.convert_to_tensor(mbbox_tensor), shape=[-1, 5 + class_num])
            lbbox_tensor = tf.reshape(tf.convert_to_tensor(lbbox_tensor), shape=[-1, 5 + class_num])
            output_tensor = tf.concat([sbbox_tensor, mbbox_tensor, lbbox_tensor], 0)
            # top_100 = tf.nn.top_k(output_tensor[:, 4], 100)
            # output_tensor = tf.gather(output_tensor, top_100.indices)
            num_tensor = tf.constant(100)


            classes_tensor = tf.argmax(output_tensor[:, 5:], axis=1)
            scores_tensor = output_tensor[:, 4]
            raw_boxs_tensor = output_tensor[:, 0:4] / IMAGE_SIZE
            print(raw_boxs_tensor.shape)
            boxs_tensor_com = raw_boxs_tensor
            print(boxs_tensor_com.shape)

            boxs_tensor_minx = raw_boxs_tensor[:, 0] - raw_boxs_tensor[:, 2] * 0.5
            boxs_tensor_miny = raw_boxs_tensor[:, 1] - raw_boxs_tensor[:, 3] * 0.5
            boxs_tensor_maxx = raw_boxs_tensor[:, 0] + raw_boxs_tensor[:, 2] * 0.5
            boxs_tensor_maxy = raw_boxs_tensor[:, 1] + raw_boxs_tensor[:, 3] * 0.5
            boxs_tensor_minx = tf.expand_dims(boxs_tensor_minx, 1)
            boxs_tensor_miny = tf.expand_dims(boxs_tensor_miny, 1)
            boxs_tensor_maxx = tf.expand_dims(boxs_tensor_maxx, 1)
            boxs_tensor_maxy = tf.expand_dims(boxs_tensor_maxy, 1)
            boxs_tensor_com = tf.concat([boxs_tensor_miny, boxs_tensor_minx, boxs_tensor_maxy, boxs_tensor_maxx], 1)


            boxs_tensor_indices = tf.image.non_max_suppression(boxs_tensor_com, scores_tensor, 100, 0.5)
            boxs_tensor_com = tf.gather(boxs_tensor_com, boxs_tensor_indices)
            classes_tensor = tf.gather(classes_tensor, boxs_tensor_indices)
            scores_tensor = tf.gather(scores_tensor, boxs_tensor_indices)



            scores_tensor_info = tf.saved_model.utils.build_tensor_info(tf.expand_dims(scores_tensor, 0))
            classes_tensor_info = tf.saved_model.utils.build_tensor_info(tf.expand_dims(classes_tensor, 0))
            boxes_tensor_info = tf.saved_model.utils.build_tensor_info(tf.expand_dims(boxs_tensor_com, 0))
            raw_boxes_tensor_info = tf.saved_model.utils.build_tensor_info(tf.expand_dims(raw_boxs_tensor, 0))
            num_tensor_info = tf.saved_model.utils.build_tensor_info(tf.expand_dims(num_tensor, 0))


            inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
            input_tensor)

            tensor_info_inputs = {
                'inputs': inputs_tensor_info
            }
            print(scores_tensor_info, inputs_tensor_info, classes_tensor_info, boxes_tensor_info, num_tensor_info)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                inputs=tensor_info_inputs,
                # outputs=tensor_info_outputs,
                outputs={
                'detection_scores': scores_tensor_info,
                'detection_classes': classes_tensor_info,
                'detection_boxes': boxes_tensor_info,
                # 'raw_boxes': raw_boxes_tensor_info,
                'num_detections': num_tensor_info,
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))


            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                tf.saved_model.signature_constants
                .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
                },
            )

            builder.save()
            print('Successfully exported model to %s' % args.output_dir)


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  image.set_shape((None, None, 3))
  image_paded = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=0)
  return image_paded/255


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()
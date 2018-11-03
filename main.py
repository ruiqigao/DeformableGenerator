from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import GeneratorDeform
import os
# Settings for assigning visible gpus
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.app.flags.FLAGS

# parameters for training
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
tf.flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')
tf.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for generator')

# parameters for model
tf.flags.DEFINE_integer('image_size', 64, 'Image size to rescale images')
tf.flags.DEFINE_float('sigma', 0.15, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('sampling_step', 20, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('step_size', 0.1, 'Step size for generator Langevin dynamics')
tf.flags.DEFINE_integer('z_app_size', 64, 'Length of latent vectors used for appearance')
tf.flags.DEFINE_integer('z_geo_size', 64, 'Length of latent vectors used for geometry')
tf.flags.DEFINE_float('geo_scale', 20, 'Scale of geometric change')

# parameters for data
tf.flags.DEFINE_string('data_dir', './data', 'The data directory')
tf.flags.DEFINE_string('category', 'face', 'The name of dataset')
tf.flags.DEFINE_boolean('prefetch', True, 'True if reading all images at once')
tf.flags.DEFINE_integer('read_len', 500, 'Number of batches per reading')

# utils
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 10, 'Number of batches to save output results')

# parameters for testing
tf.flags.DEFINE_boolean('test', True, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', 'model.ckpt-990', 'Checkpoint path to load')
tf.flags.DEFINE_integer('sample_size', 100, 'Number of images to generate during test.')


def main(_):
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
    model = GeneratorDeform(FLAGS)

    with tf.Session() as sess:
        if FLAGS.test:
            test_dir = os.path.join(output_dir, 'test')
            if tf.gfile.Exists(test_dir):
                tf.gfile.DeleteRecursively(test_dir)
            tf.gfile.MakeDirs(test_dir)
            ckpt_file = os.path.join(output_dir, 'checkpoints', FLAGS.ckpt)
            model.test(sess, ckpt_file, FLAGS.sample_size, FLAGS.batch_size, test_dir)
        else:
            sample_dir = os.path.join(output_dir, 'synthesis')
            log_dir = os.path.join(output_dir, 'log')
            model_dir = os.path.join(output_dir, 'checkpoints')
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)

            if tf.gfile.Exists(sample_dir):
                tf.gfile.DeleteRecursively(sample_dir)
            tf.gfile.MakeDirs(sample_dir)

            if tf.gfile.Exists(model_dir):
                tf.gfile.DeleteRecursively(model_dir)
            tf.gfile.MakeDirs(model_dir)

            save_dirs = {'log_dir': log_dir, 'model_dir': model_dir, 'sample_dir': sample_dir}

            model.train(sess, FLAGS, save_dirs)


if __name__ == '__main__':
    tf.app.run()

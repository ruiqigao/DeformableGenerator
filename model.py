from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from data_io import DataSet, saveSampleResults
from custom_ops import image_warp
import tensorflow as tf
import numpy as np
import scipy.io as sio


class GeneratorAppearance(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)

    def __call__(self, z, reuse=False):
        with tf.variable_scope('gen_app', reuse=reuse):
            convt0 = tf.layers.dense(z, (self.image_size // 16) * (self.image_size // 16) * 80, name='convt0',
                                                kernel_initializer=self.kernel_initializer)
            convt0 = tf.nn.relu(convt0)
            convt0 = tf.reshape(convt0, [-1, self.image_size // 16, self.image_size // 16, 80])

            convt1 = tf.layers.conv2d_transpose(convt0, 40, kernel_size=3, strides=2, padding='SAME', name='convt1',
                                                kernel_initializer=self.kernel_initializer)
            convt1 = tf.nn.relu(convt1)

            convt2 = tf.layers.conv2d_transpose(convt1, 20, kernel_size=3, strides=2, padding='SAME', name='convt2',
                                                kernel_initializer=self.kernel_initializer)
            convt2 = tf.nn.relu(convt2)

            convt3 = tf.layers.conv2d_transpose(convt2, 10, kernel_size=5, strides=2, padding='SAME', name='convt3',
                                                kernel_initializer=self.kernel_initializer)
            convt3 = tf.nn.relu(convt3)

            convt4 = tf.layers.conv2d_transpose(convt3, 3, kernel_size=5, strides=2, padding='SAME', name='convt4',
                                                kernel_initializer=self.kernel_initializer)
            convt4 = tf.nn.sigmoid(convt4)

            return convt4


class GeneratorGeometry(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)

    def __call__(self, z, reuse=False):
        with tf.variable_scope('gen_geo', reuse=reuse):
            convt0 = tf.layers.dense(z, (self.image_size // 16) * (self.image_size // 16) * 128, name='convt0',
                                                kernel_initializer=self.kernel_initializer)
            convt0 = tf.nn.relu(convt0)
            convt0 = tf.reshape(convt0, [-1, self.image_size // 16, self.image_size // 16, 128])

            convt1 = tf.layers.conv2d_transpose(convt0, 64, kernel_size=3, strides=2, padding='SAME', name='convt1',
                                                kernel_initializer=self.kernel_initializer)
            convt1 = tf.nn.relu(convt1)

            convt2 = tf.layers.conv2d_transpose(convt1, 32, kernel_size=3, strides=2, padding='SAME', name='convt2',
                                                kernel_initializer=self.kernel_initializer)
            convt2 = tf.nn.relu(convt2)

            convt3 = tf.layers.conv2d_transpose(convt2, 16, kernel_size=5, strides=2, padding='SAME', name='convt3',
                                                kernel_initializer=self.kernel_initializer)
            convt3 = tf.nn.relu(convt3)

            convt4 = tf.layers.conv2d_transpose(convt3, 2, kernel_size=5, strides=2, padding='SAME', name='convt4',
                                                kernel_initializer=self.kernel_initializer)
            convt4 = tf.nn.tanh(convt4)

            return convt4


class GeneratorDeform(object):
    def __init__(self, flags):
        self.image_size = flags.image_size
        self.learning_rate = flags.learning_rate
        self.beta1 = flags.beta1
        self.step_size = flags.step_size
        self.sigma = flags.sigma
        self.sampling_step = flags.sampling_step

        self.z_app_size, self.z_geo_size = flags.z_app_size, flags.z_geo_size
        self.geo_scale = flags.geo_scale

        self.generator_appearance = GeneratorAppearance(self.image_size)
        self.generator_geometry = GeneratorGeometry(self.image_size)
        self.obs_image = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='obs')
        self.z_app = tf.placeholder(shape=[None, self.z_app_size], dtype=tf.float32, name='z_app')
        self.z_geo = tf.placeholder(shape=[None, self.z_geo_size], dtype=tf.float32, name='z_geo')

    def build_model(self):
        self.gen_app = self.generator_appearance(self.z_app, reuse=False)
        self.gen_geo = self.generator_geometry(self.z_geo, reuse=False)
        self.gen_image = image_warp(self.gen_app, self.geo_scale * self.gen_geo)
        # self.gen_image = image_warp(self.gen_app, self.gen_geo * self.geo_scale)

        self.z_app_infer, self.z_geo_infer = self.langevin_dynamics_generator(self.z_app, self.z_geo)

        self.loss1 = tf.nn.l2_loss(self.gen_image - self.obs_image) / self.sigma / self.sigma / 100
        self.loss = tf.nn.l2_loss(self.gen_image - self.obs_image) / 100
        # self.loss_mean, self.loss_update = tf.contrib.metrics.streaming_mean(self.loss1)

        optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)
        trainable_vars = tf.trainable_variables()
        self.apply_grads = optim.minimize(self.loss1, var_list=trainable_vars)

        # symbolic langevins

        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def langevin_dynamics_generator(self, z_app, z_geo):
        def cond(i, z_app, z_geo):
            return tf.less(i, self.sampling_step)

        def body(i, z_app, z_geo):
            noise_app = tf.random_normal(shape=tf.shape(z_app), name='noise_app')
            noise_geo = tf.random_normal(shape=tf.shape(z_geo), name='noise_geo')
            gen_app = self.generator_appearance(z_app, reuse=True)
            gen_geo = self.generator_geometry(z_geo, reuse=True)
            gen_image = image_warp(gen_app, self.geo_scale * gen_geo)
            # gen_image = image_warp(gen_app, gen_geo * self.geo_scale)
            loss = (tf.nn.l2_loss(gen_image - self.obs_image) / self.sigma / self.sigma +
                    100 * tf.nn.l2_loss(z_app) + 100 * tf.nn.l2_loss(z_geo)) / 100

            grad_app = tf.gradients(loss, z_app, name='grad_app')[0]
            z_app = z_app - 0.5 * self.step_size * self.step_size * grad_app + self.step_size * noise_app
            grad_geo = tf.gradients(loss, z_geo, name='grad_geo')[0]
            z_geo = z_geo - 0.5 * self.step_size * self.step_size * grad_geo + self.step_size * noise_geo

            return tf.add(i, 1), z_app, z_geo

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z_app, z_geo = tf.while_loop(cond, body, [i, z_app, z_geo])
            return z_app, z_geo

    def train(self, sess, flags, save_dirs):
        self.build_model()

        # Prepare training data
        data_path = os.path.join(flags.data_dir, flags.category)
        dataset = DataSet(data_path, image_size=self.image_size, batch_sz=flags.batch_size,
                          prefetch=flags.prefetch, read_len=flags.read_len)

        # initialize training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=20)
        writer = tf.summary.FileWriter(save_dirs['log_dir'], sess.graph)

        # make graph immutable
        tf.get_default_graph().finalize()

        # store graph in protobuf
        with open(save_dirs['model_dir'] + '/graph.proto', 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))

        # train
        inception_mean, inception_sd = [], []

        z_app = np.random.randn(len(dataset), self.z_app_size)
        z_geo = np.random.randn(len(dataset), self.z_geo_size)
        for epoch in range(flags.num_epochs):
            start_time = time.time()
            loss_avg = []
            for i in range(dataset.num_batch):
                batch_idx = np.arange(i * flags.batch_size, (i+1) * flags.batch_size)
                obs_data = dataset.images[batch_idx]
                z_app_batch = z_app[batch_idx]
                z_geo_batch = z_geo[batch_idx]

                # infer z
                feed_dict = {self.obs_image: obs_data, self.z_app: z_app_batch, self.z_geo: z_geo_batch}
                z_app_batch_infer, z_geo_batch_infer = sess.run([self.z_app_infer, self.z_geo_infer], feed_dict=feed_dict)
                z_app[batch_idx] = z_app_batch_infer
                z_geo[batch_idx] = z_geo_batch_infer

                # update weights
                feed_dict = {self.obs_image: obs_data, self.z_app: z_app_batch_infer, self.z_geo: z_geo_batch_infer}
                gen_app, gen_image, gen_geo, loss, _ = \
                    sess.run([self.gen_app, self.gen_image, self.gen_geo, self.loss, self.apply_grads], feed_dict=feed_dict)

                loss_avg.append(loss)

            loss_avg = np.mean(np.asarray(loss_avg))

            if epoch % flags.log_step == 0:
                saver.save(sess, '%s/%s' % (save_dirs['model_dir'], 'model.ckpt'), global_step=epoch)
                saveSampleResults(gen_app, '%s/%d_app.png' % (save_dirs['sample_dir'], epoch), col_num=10, margin_syn=2)
                saveSampleResults(gen_image, '%s/%d_img.png' % (save_dirs['sample_dir'], epoch), col_num=10, margin_syn=2)
                saveSampleResults(obs_data, '%s/%d_obs.png' % (save_dirs['sample_dir'], epoch), col_num=10, margin_syn=2)

            end_time = time.time()
            print('Epoch %d, loss: %4f, time: %.2f' % (epoch, loss_avg, end_time - start_time))

    def test(self, sess, ckpt, sample_size, batch_size, test_dir):
        assert ckpt is not None, 'no checkpoint provided.'
        gen_app = self.generator_appearance(self.z_app)
        gen_geo = self.generator_geometry(self.z_geo) * self.geo_scale
        gen_image = image_warp(gen_app, gen_geo)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        num_batches = int(np.ceil(sample_size / batch_size))

        sample_results = np.zeros(shape=(num_batches * batch_size, self.image_size, self.image_size, 3))
        for i in range(num_batches):
            z_geo_value = np.random.randn(batch_size, self.z_geo_size)
            z_app_value = np.random.randn(batch_size, self.z_app_size)
            gen_app_value, gen_image_value = sess.run([gen_app, gen_image], feed_dict={self.z_geo: z_geo_value, self.z_app: z_app_value})
            sample_results[i * batch_size:(i + 1) * batch_size] = gen_image_value

            if i % 10 == 0:
                print("Sampling batches: {}, from {} to {}".format(i, i * batch_size,
                                                                   min((i+1) * batch_size, sample_size)))
            saveSampleResults(gen_app_value, '%s/batch%d_app.png' % (test_dir, i))
            saveSampleResults(gen_image_value, '%s/batch%d_img.png' % (test_dir, i))
        sample_results = sample_results[:sample_size]
        sample_results = np.minimum(1, np.maximum(0, sample_results)) * 255
        sio.savemat('%s/samples.mat' % test_dir, {'samples': sample_results})

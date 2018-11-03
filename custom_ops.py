from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# def image_warp(im, flow):
#     """Performs a backward warp of an image using the predicted flow.
#     Args:
#         im: Batch of images. [num_batch, height, width, channels]
#         flow: Batch of flow vectors. [num_batch, height, width, 2]
#     Returns:
#         warped: transformed image of the same shape as the input image.
#     """
#     with tf.variable_scope('image_warp'):
#
#         num_batch, height, width, channels = tf.unstack(tf.shape(im))
#         max_x = tf.cast(width - 1, 'int32')
#         max_y = tf.cast(height - 1, 'int32')
#         zero = tf.zeros([], dtype='int32')
#
#         # We have to flatten our tensors to vectorize the interpolation
#         im_flat = tf.reshape(im, [-1, channels])
#         flow_flat = tf.reshape(flow, [-1, 2])
#
#         # Floor the flow, as the final indices are integers
#         # The fractional part is used to control the bilinear interpolation.
#         flow_floor = tf.to_int32(tf.floor(flow_flat))
#         bilinear_weights = flow_flat - tf.floor(flow_flat)
#
#         # Construct base indices which are displaced with the flow
#         pos_x = tf.tile(tf.range(width), [height * num_batch])
#         grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
#         pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])
#
#         x = flow_floor[:, 0]
#         y = flow_floor[:, 1]
#         xw = bilinear_weights[:, 0]
#         yw = bilinear_weights[:, 1]
#
#         # Compute interpolation weights for 4 adjacent pixels
#         # expand to num_batch * height * width x 1 for broadcasting in add_n below
#         wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
#         wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
#         wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
#         wd = tf.expand_dims(xw * yw, 1) # bottom right pixel
#
#         x0 = pos_x + x
#         x1 = x0 + 1
#         y0 = pos_y + y
#         y1 = y0 + 1
#
#         x0 = tf.clip_by_value(x0, zero, max_x)
#         x1 = tf.clip_by_value(x1, zero, max_x)
#         y0 = tf.clip_by_value(y0, zero, max_y)
#         y1 = tf.clip_by_value(y1, zero, max_y)
#
#         dim1 = width * height
#         batch_offsets = tf.range(num_batch) * dim1
#         base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
#         base = tf.reshape(base_grid, [-1])
#
#         base_y0 = base + y0 * width
#         base_y1 = base + y1 * width
#         idx_a = base_y0 + x0
#         idx_b = base_y1 + x0
#         idx_c = base_y0 + x1
#         idx_d = base_y1 + x1
#
#         Ia = tf.gather(im_flat, idx_a)
#         Ib = tf.gather(im_flat, idx_b)
#         Ic = tf.gather(im_flat, idx_c)
#         Id = tf.gather(im_flat, idx_d)
#
#         warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
#         warped = tf.reshape(warped_flat, [num_batch, height, width, channels])
#
#     return warped


def _meshgrid(num_batch, height, width):
    with tf.variable_scope('mymeshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        # ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat])
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))
        grid = tf.reshape(grid, tf.stack([num_batch, 2, -1]))
        return grid


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])


def _interpolate(im, x, y, xd, yd, out_size):
    with tf.variable_scope('myinterpolate'):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        xd = tf.cast(xd, 'float32')
        yd = tf.cast(yd, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]

        zero1 = tf.zeros([], dtype='float32')
        max_y1 = tf.cast(tf.shape(im)[1] - 1, 'float32')
        max_x1 = tf.cast(tf.shape(im)[2] - 1, 'float32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0
        ###################
        x = x + xd
        y = y + yd
        x = tf.clip_by_value(x, zero1 + 0.000001, max_x1 - 0.000001)
        y = tf.clip_by_value(y, zero1 + 0.000001, max_y1 - 0.000001)
        ##################
        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output


def image_warp(im, dlm):
    num_batch, height, width, channels = tf.unstack(tf.shape(im))
    dlm = tf.reshape(dlm, (-1, height * width, 2))
    dlm = tf.transpose(dlm, [0, 2, 1])
    grid = _meshgrid(num_batch, height, width)
    x_s = tf.slice(grid, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(grid, [0, 1, 0], [-1, 1, -1])
    x_s_flat = tf.reshape(x_s, [-1])
    y_s_flat = tf.reshape(y_s, [-1])
    xd_s = tf.slice(dlm, [0, 0, 0], [-1, 1, -1])
    yd_s = tf.slice(dlm, [0, 1, 0], [-1, 1, -1])
    xd_s_flat = tf.reshape(xd_s, [-1])
    yd_s_flat = tf.reshape(yd_s, [-1])
    input_transformed = _interpolate(
        im, x_s_flat, y_s_flat, xd_s_flat, yd_s_flat,
        (height, width))
    Yhat = tf.reshape(
        input_transformed, tf.stack([num_batch, height, width, channels]))
    return Yhat





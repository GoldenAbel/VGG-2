# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/roshan/class_models/CIFAR10/CIFAR10_Data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY_1 = 75.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.01 # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
WD = 0.001                        # Weight decay (Regularization)

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin') 
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def conv_op(input_op, name, kw, kh, n_out, dw, dh):
	n_in = input_op.get_shape()[-1].value

	with tf.variable_scope(name) as scope:
		kernel = _variable_on_cpu('weights',
		                         shape=[kh, kw, n_in, n_out], 
		                         initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
		biases = _variable_on_cpu('bias', [n_out], tf.constant_initializer(0.0))
		z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())  
		activation = tf.nn.relu(z, name=scope.name)
	return activation

def fc_op(input_op, name, n_out):
	n_in = input_op.get_shape()[-1].value

	with tf.variable_scope(name) as scope:
	  kernel = _variable_on_cpu('weights',
	                           shape=[n_in, n_out],
	                           initializer=tf.contrib.layers.xavier_initializer())
	  biases = _variable_on_cpu('biases', [n_out], tf.constant_initializer(0.1))
	  activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope.name)
	return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
  return tf.nn.max_pool(input_op,
                        ksize=[1, kh, kw, 1],
                        strides=[1, dh, dw, 1],
                        padding='SAME',
                        name=name)


def inference(images, training=True):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  dropout_keep_prob = 0.7 if training else 1.0
  # assume input_op shape is 224x224x3

  # block 1 -- outputs 112x112x64
  conv1_1 = conv_op(images, name="conv1_1", kh=3, kw=3, n_out=16, dh=1, dw=1)
  conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=16, dh=1, dw=1)
  pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)

  # block 2 -- outputs 56x56x128
  conv2_1 = tf.nn.dropout(conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=32, dh=1, dw=1), dropout_keep_prob)
  conv2_2 = tf.nn.dropout(conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1), dropout_keep_prob)
  pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

  # # block 3 -- outputs 28x28x256
  conv3_1 = tf.nn.dropout(conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1), dropout_keep_prob)
  conv3_2 = tf.nn.dropout(conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1), dropout_keep_prob)
  pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)

  # block 4 -- outputs 14x14x512
  conv4_1 = tf.nn.dropout(conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
  conv4_2 = tf.nn.dropout(conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
  conv4_3 = tf.nn.dropout(conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
  pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)


  # flatten
  shp = pool4.get_shape()
  flattened_shape = shp[1].value * shp[2].value * shp[3].value
  resh1 = tf.reshape(pool4, [-1, flattened_shape], name="resh1")

  # fully connected
  fc6 = fc_op(resh1, name="fc6", n_out=1024)
  fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")

  fc7 = fc_op(fc6_drop, name="fc7", n_out=1024)
  fc7_drop = tf.nn.dropout(fc7, dropout_keep_prob, name="fc7_drop")

  fc8 = fc_op(fc7_drop, name="fc8", n_out=10)
  
  softmax = tf.nn.softmax(fc8)
  predictions = tf.argmax(softmax, 1)
  return softmax, predictions, fc8
  

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY_1)


  # Decay the learning rate exponentially based on the number of steps.
  decayed_learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
                                                                     
  lr=decayed_learning_rate                                
                                                                       
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(lr,0.9)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def evaluate_op(predictions, labels):
  """Evaluate the quality of the predictions at predicting the label.
  Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
  Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(predictions, labels, 1)

  # Return the number of true entries.
  total_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  return accuracy, total_correct
 
 
def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

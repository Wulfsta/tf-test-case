import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import argparse

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications import vgg19

from tensorflow.python.keras import models

import time
import IPython
from PIL import Image
import IPython.display


content_layers = [
    'block5_conv2',
]

num_content_layers = len(content_layers)

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1',
]

num_style_layers = len(style_layers)


def preprocess_img(img_path):
  # Set the proportions of the image.
  
  width, height = load_img(img_path).size
  img_height = 500
  img_width = int(width * img_height / height)
  
  img = load_img(img_path, target_size=(img_height, img_width))
  img = img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = vgg19.preprocess_input(img)
  
  return img


def deprocess_img(processed_img):
  x = processed_img.copy()
  
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)

  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x


def get_model():
  """Creates a model with access to intermediate layers. 
  
  These layers will then be used to create a new model that will take the
  content image and return the outputs from these intermediate layers from the
  VGG model. 
  
  Returns:
    A keras model that takes image inputs and outputs the style and content
    intermediate layers.
  """

  vgg = vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs

  return models.Model(vgg.input, model_outputs)


def compute_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)


def compute_style_loss(base_style, gram_target):
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))


def feature_representations(model, content_path, style_path):
  """Helper function to compute our content and style feature representations.

  This function will simply load and preprocess both the content and style 
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: the model that we are using
    content_path: the path to the content image
    style_path: the path to the style image
    
  Returns:
    The style and content features.
  """

  content_img = preprocess_img(content_path)
  style_img = preprocess_img(style_path)

  style_outputs = model(style_img)
  content_outputs = model(content_img)
  
  
  style_features = [
      style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [
      content_layer[0] for content_layer in content_outputs[num_style_layers:]]

  return style_features, content_features


def compute_loss(
    model, loss_weights, init_img, gram_style_features, content_features):
  """Computes the total loss.
  
  Arguments:
    model: the model that will give us access to the intermediate layers
    loss_weights: the weights of each contribution of each loss function
      (style weight, content weight, and total variation weight)
    init_img: the initial base image, that is updated according to the
      optimization process
    gram_style_features: precomputed gram matrices corresponding to the 
      defined style layers of interest
    content_features: precomputed outputs from defined content layers of
      interest
      
  Returns:
    The total loss, style loss, content loss, and total variational loss.
  """
  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers.
  model_outputs = model(init_img)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_loss, content_loss = 0, 0

  # Accumulate style losses from all layers. All weights are equal.
  style_layer_weight = 1.0 / num_style_layers

  for target_style, generated_style in zip(
      gram_style_features, style_output_features):
    style_loss += style_layer_weight * compute_style_loss(
        generated_style[0], target_style)
    
  # Accumulate content losses from all layers. All weights are equal.
  content_layer_weight = 1.0 / num_content_layers
  for target_content, generated_content in zip(
      content_features, content_output_features):
    content_loss += content_layer_weight * compute_content_loss(
        generated_content[0], target_content)
  
  style_loss *= style_weight
  content_loss *= content_weight

  total_loss = style_loss + content_loss 

  return total_loss, style_loss, content_loss


def compute_gradients(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)

  total_loss = all_loss[0]

  return tape.gradient(total_loss, cfg['init_img']), all_loss


def run_style_transfer(content_path, style_path, n_iterations=1000,
                       content_weight=1e4, style_weight=1e-4,
                       display_iterations=True):
  """Run the neural style transfer algorithm.
  
  Arguments:
    content_path: the filename of the target content image
    style_path: the filename of the reference style image
    content_weight: the weight for the content features, where higher means the
      generated image will put heavier emphasis on content (default 1e-4)
    style_weight: the weight for the style features, where higher means the
      generated image put heavier emphasis on style (default 1e4)
    n_iterations: the number of optimization iterations (default 1000)
    display_iterations: whether to display intermediate iterations of the
      generated images (default True)
    
  Returns:
    The final generated image and the total loss for that image.
  """

  model = get_model() 
  
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  for layer in model.layers:
    layer.trainable = False
  
  style_features, content_features = feature_representations(
      model, content_path, style_path)

  gram_style_features = [
      gram_matrix(style_feature) for style_feature in style_features
  ]
  
  init_img = preprocess_img(content_path)
  init_img = tf.Variable(init_img, dtype=tf.float32)

  # The optimizer params are somewhat arbitrary.
  # See tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#__init__
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
  
  # Store the result that minimizes loss as the best one.
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model':               model,
      'loss_weights':        loss_weights,
      'init_img':            init_img,
      'gram_style_features': gram_style_features,
      'content_features':    content_features
  }

  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   

  imgs = []
  for i in range(n_iterations):
    
    gradients, all_loss = compute_gradients(cfg)
    total_loss, style_loss, content_loss = all_loss
    opt.apply_gradients([(gradients, init_img)])
    clipped = tf.clip_by_value(init_img, min_vals, max_vals)
    init_img.assign(clipped)
    end_time = time.time() 
    
    # Update best loss and best image from total loss. 
    if total_loss < best_loss:
      best_loss = total_loss
      best_img = deprocess_img(init_img.numpy())
      
    if display_iterations:
      
      n_rows, n_cols = 2, 5
      display_interval = n_iterations / (n_rows * n_cols)
  
      if i % display_interval == 0:
        start_time = time.time()

        plot_img = deprocess_img(init_img.numpy())
        imgs.append(plot_img)

        IPython.display.clear_output(wait=True)
        IPython.display.display_png(Image.fromarray(plot_img))

        print('Iteration: {}'.format(i))        
        print('Total loss: {:.4e}, ' 
              'style loss: {:.4e}, '
              'content loss: {:.4e}, '
              'time: {:.4f}s'.format(total_loss, style_loss, content_loss,
                                     time.time() - start_time))

  if display_iterations:
    IPython.display.clear_output(wait=True)

    plt.figure(figsize=(14,4))

    for i,img in enumerate(imgs):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        plt.axis('off')
    
    print('Total time: {:.4f}s'.format(time.time() - global_start))
      
  return best_img, best_loss


def main():
    best_img, best_loss = run_style_transfer('eminem.jpg', 'fractal.jpg')
    print(best_loss.numpy())

    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.axis('off')

    plt.show()
    


if __name__=='__main__':
    main()

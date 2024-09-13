#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
import math
from matplotlib import pyplot as plt, cm
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import img_to_array
import tensorflow.keras.activations as tf_act

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.scorecam import Scorecam

from tensorflow.keras.applications.resnet50 import (ResNet50,
                                                    preprocess_input as resnet50_preprocess_input,
                                                    decode_predictions as resnet50_decode_predictions)

import tensorflow_addons as tfa


# In[2]:


SCALE_MIN = 0.3
SCALE_MAX = 0.3
rotate_max = np.pi/8 # 22.5 degrees in either direction

MAX_ROTATION = 22.5

# os.chdir('..') # UNCOMMENT THIS IF USING FROM root/utils DIRECTORY
ROOT_DIR = os.getcwd()

model = ResNet50(weights='imagenet')

# Download the json file of list of classes in imagenet with index
if os.path.isfile('imagenet_class_index.json') == False:
    os.system('wget "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"')

CLASS_INDEX = json.load(open("imagenet_class_index.json"))

classlabel = []
for i_dict in range(len(CLASS_INDEX)):
    classlabel.append(CLASS_INDEX[str(i_dict)][1])

IOU = []


# In[3]:


def circle_mask(shape, sharpness = 40):
    """Return a circular mask of a given shape"""
    
    assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

    diameter = shape[0]  
    x = np.linspace(-1, 1, diameter)
    y = np.linspace(-1, 1, diameter)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx**2 + yy**2) ** sharpness
    mask = 1 - np.clip(z, -1, 1)
    # plt.contour(x, y, z)
    # plt.imshow(mask)

    # plt.xlim((-3, 3))
    # plt.ylim(-3, 3)
    mask = np.expand_dims(mask, axis=2)
    mask = np.broadcast_to(mask, shape).astype(np.float32)
    return mask

def show(im):
    plt.axis('off')
    plt.imshow(im, interpolation="nearest")
    plt.show()

def transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
    
    """
    If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
    then it maps the output point (x, y) to a transformed input point 
    (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
    where k = c0 x + c1 y + 1. 
    The transforms are inverted compared to the transform mapping input points to output points.
    """

    rot = float(rot_in_degrees) / 90. * (math.pi/2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
      [[math.cos(-rot), -math.sin(-rot)],
      [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale
    xform_matrix = rot_matrix * inv_scale

    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image. 
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2

    x_origin_shifted, y_origin_shifted = np.matmul(
      xform_matrix,
      np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift/(2*im_scale))
    b2 = y_origin_delta - (y_shift/(2*im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

def random_overlay(imgs, patch, image_shape, batch_size):

    """Augment images with random rotation, transformation.

    Image: BATCHx299x299x3
    Patch: 50x50x3

    """
    # Add padding
    
    masked_img = np.zeros((224, 224, 3), dtype=np.float32)
    masked_patch = np.ones((224, 224, 3), dtype=np.float32)

    image_mask = circle_mask(image_shape)
    masked_image_mask = circle_mask(image_shape)

    image_mask = tf.stack([image_mask] * batch_size)
    padded_patch = tf.stack([patch] * batch_size)

    masked_image_mask = tf.stack([masked_image_mask] * batch_size)
    masked_padded_patch = tf.stack([masked_patch] * batch_size)

    transform_vecs = []    

    def random_transformation(scale_min, scale_max, width):
        im_scale = np.random.uniform(low=scale_min, high=scale_max)

        padding_after_scaling = (1-im_scale) * width
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)


        rot = np.random.uniform(-MAX_ROTATION, MAX_ROTATION)
        
        return transform_vector(width,
                                x_shift=x_delta,
                                y_shift=y_delta,
                                im_scale=im_scale, 
                                rot_in_degrees=rot)    

    for i in range(batch_size):
        # Shift and scale the patch for each image in the batch
        random_xform_vector = tf.numpy_function(random_transformation, [SCALE_MIN, SCALE_MAX, image_shape[0]], tf.float32)
        random_xform_vector.set_shape([8])

        transform_vecs.append(random_xform_vector)

    image_mask = tfa.image.transform(image_mask, transform_vecs, "BILINEAR")
    padded_patch = tfa.image.transform(padded_patch, transform_vecs, "BILINEAR")
    
    
    masked_image_mask = tfa.image.transform(masked_image_mask, transform_vecs, "BILINEAR")
    masked_padded_patch = tfa.image.transform(masked_padded_patch, transform_vecs, "BILINEAR")
        
    inverted_mask = (1 - image_mask)
    return (imgs * inverted_mask + padded_patch * image_mask), (masked_img * inverted_mask + masked_padded_patch * masked_image_mask)


# In[4]:


def get_binary_mask(image, adv_class_lbl, adv_predictions):

    def model_modifier_function(cloned_model):
        cloned_model.layers[-1].activation = tf_act.linear

    score = CategoricalScore([adv_class_lbl])

    scorecam = Scorecam(model, model_modifier=model_modifier_function)

    cam = scorecam(score, np.float32(image).copy(), penultimate_layer=-1, max_N=10)

    heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)
    return cam


# In[5]:


def IoU(true_mask, pred_mask):
    true_mask_area = np.count_nonzero(true_mask == 1)       # I assume this is faster as mask1 == 1 is a bool array
    pred_mask_area = np.count_nonzero(pred_mask == 1)
    intersection = np.count_nonzero( np.logical_and( true_mask, pred_mask) )
    iou = intersection / (true_mask_area + pred_mask_area - intersection)
    return iou


# In[6]:


def process_image(clean_image_gen, clean_img, clean_label, plot=False):
    img = img_to_array(clean_img) # .astype(np.uint8)

    adv_trans_image, random_patch_mask = random_overlay(imgs=img, patch=patch_img, image_shape=[224, 224, 3], batch_size=1)
    adv_trans_image = adv_trans_image.numpy().astype(np.uint8)
    random_patch_mask = random_patch_mask[0, :, :, 0]
    
    plt.show()
    tmp_img = resnet50_preprocess_input(adv_trans_image.copy())
    img_pred = model.predict(tmp_img)
    prediction = resnet50_decode_predictions(img_pred, top=1)
    
    adv_prediction = prediction[0][0]
    adv_class_lbl = classlabel.index(prediction[0][0][1])
    scorecam_mask = get_binary_mask(adv_trans_image, adv_class_lbl, adv_prediction)

    scorecam_mask = np.where(scorecam_mask >= 0.5, 1., 0.)

#     fig, axes = plt.subplots(1, 3)
#     fig.tight_layout()
    
#     axes[0].imshow(adv_trans_image[0])
#     axes[0].set_title('adv_trans_image')
#     axes[0].axis('off')
    
#     axes[1].imshow(random_patch_mask, cmap='gray')
#     axes[1].set_title('random_patch_mask')
#     axes[1].axis('off')
    
#     axes[2].imshow(scorecam_mask[0], cmap='gray')
#     axes[2].set_title('scorecam_mask')
#     axes[2].axis('off')

    iou = IoU(random_patch_mask, scorecam_mask)
    IOU.append(iou)


# In[7]:


os.chdir(ROOT_DIR)
os.getcwd()


# In[ ]:


os.chdir(ROOT_DIR)

# Set the DATA DIR
clean_dir = r'resnet_40\clean_images'

# Load the patch image
patch_img = load_img(os.path.join(r'patch\resnet50_patch.png'), target_size=(224, 224, 3), interpolation='nearest')
patch_img = img_to_array(patch_img)

# class name and folder name of the image that we are reading
class_name = []
file_name = []

# os.chdir(clean_dir)
# for folder in os.listdir():
clean_data_directory = os.path.join(os.getcwd(), clean_dir)
clean_images_gen = image_dataset_from_directory(
    clean_data_directory,
    seed=42, 
    image_size=(224, 224),
    batch_size=50, # The dataset will yield individual samples.
    color_mode='rgb',
    shuffle=False)
i =1
for clean_img, clean_label in clean_images_gen.take(50):
    [process_image(clean_images_gen, clean_img, clean_lbl) for clean_img, clean_lbl in zip(clean_img, clean_label)]

meanIoU = np.mean(IOU)

filename = f'IoU_resnet50'

with open(filename, 'w')as f:
    f.write(f'IoU: {meanIoU}')

    print(meanIoU)


# In[ ]:


print(np.mean(IOU))

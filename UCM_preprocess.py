import numpy as np
import os
import random
from PIL import Image
import tensorflow as tf
#import cv2
import matplotlib.pyplot as plt
import Image
# UCM_PATH = '/home/hpc-126/remote-host/UCM/UCMjpg'
# Imagesize = 32
# Batch_size =4
def process_data(path,Imagesize):
    image = []
    label = []
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    for root, dirs, files in os.walk(train_path):
        if root == path:
            continue
        label_name = os.path.basename(root)
        for name in files:
            im = Image.open(os.path.join(root, name))
            (w, h) = im.size
            if w != Imagesize or h != Imagesize:
                im = im.resize((Imagesize,Imagesize), Image.ANTIALIAS)
                im.save(os.path.join(root, name))
            image.append([os.path.join(root, name),label_name])

    image = np.array(image)
    np.random.shuffle(image)
    # test_set = image[1728:]
    train_set = image
    print ('train set length is:')
    print (train_set.shape)
    image = []
    for root, dirs, files in os.walk(test_path):
        if root == path:
            continue
        label_name = os.path.basename(root)
        for name in files:
            im = Image.open(os.path.join(root, name))
            (w, h) = im.size
            if w != Imagesize or h != Imagesize:
                im = im.resize((Imagesize,Imagesize), Image.ANTIALIAS)
                im.save(os.path.join(root, name))
            image.append([os.path.join(root, name),label_name])

    image = np.array(image)
    #np.random.shuffle(image)
    # test_set = image[1728:]
    test_set = image
    print ('test set size is :')
    print (test_set.shape)
    return  train_set,test_set

# def UCM_generator(train_set,Batch_size,Imagesize):
#
#     image_list = list(train_set[:,0])
#     label_list = list(train_set[:,1])
#     data = np.zeros((len(image_list),Imagesize,Imagesize,3),np.uint8)
#     for idx in range(len(image_list)):
#         img = cv2.imread(image_list[idx])
#         data[idx,:,:,:]
#     label_list = tf.string_to_number(label_list)
#     image_list = tf.cast(image_list,tf.string)
#     label_list =tf.cast(label_list,tf.int32)
#     input_queue = tf.train.slice_input_producer([image_list,label_list])
#     label = input_queue[1]
#     image_contents = tf.read_file(input_queue[0])
#     image = tf.image.decode_jpeg(image_contents,channels=3)
#     # crop and padding the images
#     image = tf.image.resize_image_with_crop_or_pad(image, Imagesize,Imagesize)
#
#     #
# #    image = tf.image.per_image_standardization(image)
#     image_batch,label_batch = tf.train.batch([image,label],batch_size=Batch_size,num_threads=16,capacity=1680)
#
#     label_batch = tf.reshape(label_batch,[Batch_size])
#     return image_batch,label_batch


def datagenerator(dataset,Batch_size,Imagesize):
    images=[]
    image_list = list(dataset[:, 0])
    label_list = list(dataset[:, 1])
    count =0
    for filename in image_list:
        image = Image.open(filename)
        images.append(np.array(image))
    images = np.array(images)
    images = images.transpose(0,3,1,2)
    images = images.reshape(len(image_list),Imagesize*Imagesize*3)
    label_list = [int(i) for i in label_list]
    label_list = np.array(label_list)
    print label_list

    def get_epoch():
        for i in xrange(len(images) / Batch_size):
            yield (images[i*Batch_size:(i+1)*Batch_size], label_list[i*Batch_size:(i+1)*Batch_size])
    return get_epoch

def load (PATH,Batch_size,Imagesize):

    train_set, test_set = process_data(PATH, Imagesize)

    return (datagenerator(train_set,Batch_size,Imagesize),
            datagenerator(test_set,Batch_size,Imagesize))
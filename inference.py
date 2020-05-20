import os
import glob
import argparse
import datetime

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD

from model import Autoencoder
import utils.load_data as dataloader


def build_model():
    '''Reconstruct a trined model from saved data.
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    with open("models/2020-03-06-16-21-20/topology.txt", "r") as topology:
        num_filters = tuple(map(int, topology.readline()[1:-1].split(', ')))

    input_shape = (None, None, 3)
   
    model = Autoencoder(input_shape=input_shape, num_filters=num_filters)
    model = model.build()

    model.load_weights("models/2020-03-06-16-21-20/weights.h5")
    model.compile(optimizer="adam", loss="MSE", metrics=["accuracy"])

    return model



def get_mask_of_foreign_obj(inp_img, out_img, metrics="l2"):
    '''Compute mask of a foreign object in a scene (if there is one).
    '''
    
    inp_img = (inp_img * 255).astype(int)
    out_img = (out_img * 255).astype(int)

    if metrics == "l1":
    # Count mask using L1 metric

        mask = np.absolute(inp_img - out_img).astype(np.uint8)
        mask = mask > 30 # difference threshold

        # add pixel to mask if at least one of the 3 channels is true
        mask = np.logical_or(mask[:,:,0], np.logical_or(mask[:,:,1], mask[:,:,2]))
        mask = mask.astype(np.uint8)

        # get rid of offset mask pixels using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20, 20))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask


    if metrics == "l2":
    # Count mask using L2 metric (Euclidian distance)

        mask = np.sqrt((inp_img - out_img) ** 2).astype(np.uint8) # L2 difference
        mask = mask > 30 # difference threshold

        # add pixel to mask if at least one of the 3 channels is true
        mask = np.logical_or(mask[:,:,0], np.logical_or(mask[:,:,1], mask[:,:,2]))
        mask = mask.astype(np.uint8)

        # # get rid of offset mask pixels using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20, 20))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    elif metrics == "ssim":
    # Count mask using Structure similarity index
        _, mask = ssim(inp_img, out_img, win_size=49, multichannel=True, full=True)
        mask = (mask < 0.9).astype(np.uint8)

        plt.imshow(mask)
        plt.show()

        return mask

    else:
        raise ValueError("Unknown similarity metric: {}".format(metrics))
    



def find_foreign_object(inp_img, out_img, return_is_fod=False):
    ''' Find mask of a foreign object in a scene and compute its bounding box.
        Return the input image with the bounding box applied.
    '''

    mask = get_mask_of_foreign_obj(inp_img, out_img)
    image = (inp_img * 255).astype(np.uint8)

    if np.any(mask):

        x, y, w, h = cv2.boundingRect(mask)
        result = cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 15)
        result = result.astype(np.uint8)

        if return_is_fod:
            return result, 1
        else:
            return result

    else:
        if return_is_fod:
            return image.astype(np.uint8), 0
        else:
            return image.astype(np.uint8)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", default="0")
    parser.add_argument("--file_dir", default="dataset_test")
    parser.add_argument("--topology", default="models/2020-03-06-16-21-20/topology.txt")
    parser.add_argument("--weights", default="models/2020-03-06-16-21-20/weights.h5")

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    with open(args.topology, "r") as topology:
        num_filters = tuple(map(int, topology.readline()[1:-1].split(', ')))
    
    files = glob.glob(os.path.join(args.file_dir, "*.png"))

    input_shape = (None, None, 3)
   
    # Reconstruct model from saved weights
    model = Autoencoder(input_shape=input_shape, num_filters=num_filters)
    model = model.build()
    print(model.summary())

    model.load_weights(args.weights)
    model.compile(optimizer="adam", loss="MSE", metrics=["accuracy"])

    # Generate time stamp for unique id of the result
    time_stamp = "{date:%Y-%m-%d-%H-%M-%S}".format(date=datetime.datetime.now())

    # Pass images to network
    for file, i in zip(files, range(len(files))):

        inp_img = cv2.imread(file) / 255
        inp_img = np.expand_dims(inp_img, axis=0)

        out_img = model.predict(inp_img)

        inp_img = np.squeeze(inp_img, axis=0)
        out_img = np.squeeze(out_img, axis=0)

        # Find bounding box of foreign object
        res = find_foreign_object(inp_img, out_img)

        inp_img = cv2.cvtColor((inp_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        f = plt.figure()
        f.add_subplot(1, 2 , 1)
        plt.imshow(inp_img)
        f.add_subplot(1, 2 , 2)
        plt.imshow(res)
        plt.show(block=True)

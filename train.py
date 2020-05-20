import os
import glob
import argparse
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD

from model import Autoencoder
from inference import get_mask_of_foreign_obj
import utils.load_data as dataloader



def jaccard_index(mask, gt):

    if np.any(mask):
        x1, y1, w1, h1 = cv2.boundingRect(mask)
        box1 = cv2.rectangle(np.zeros((2048, 2048)), (x1, y1), (x1+w1, y1+h1), 1, -1)
    else:
        box1 = None

    if np.any(gt):
        x2, y2, w2, h2 = cv2.boundingRect(gt)
        box2 = cv2.rectangle(np.zeros((2048, 2048)), (x2, y2), (x2+w2, y2+h2), 1, -1)
    else:
        box2 = None

    if box1 is not None and box2 is not None:
        intersection = np.logical_and(box1, box2)
        union = np.logical_or(box1, box2)

        j = np.count_nonzero(intersection) / np.count_nonzero(union)

    elif box1 is None and box2 is None:
        j = 1

    else:
        j = 0
    
    return j



def callbacks(model_path, test_ds, test_gt):
    '''
    Build callbacks for fit function. 
    '''
    # Save best model after each epoch
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=False, save_weights_only=True)

    # Track training progress with TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Compute bounding box of the foreign object a print its accuracy
    class PlotOutput(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):

            test_out = model.predict(next(iter(test_ds)))
            accuracy = [0, 0]

            for inp, out, gt, i in zip(test_ds.unbatch(), test_out, test_gt.unbatch(), range(test_out.shape[0])):
                inp_img = inp.numpy() * 255
                out_img = out * 255

                mask = get_mask_of_foreign_obj(inp_img, out_img)
                mask_gt = np.squeeze(gt.numpy().astype(np.uint8))

                # Rate accuracy of the bounding box by jaccard index
                j = jaccard_index(mask, mask_gt)

                accuracy[0] += j
                accuracy[1] += 1

                if epoch == 0:
                    cv2.imwrite(os.path.join(img_dir, "fig-{}-in.png".format(i)), inp_img)
                    cv2.imwrite(os.path.join(img_dir, "fig-{}-mask_gt.png".format(i)), mask_gt)
                cv2.imwrite(os.path.join(img_dir, "ep-{}-fig-{}-out.png".format(epoch, i)), out_img)
                cv2.imwrite(os.path.join(img_dir, "ep-{}-fig-{}-mask.png".format(epoch, i)), mask)

            average_accuracy = accuracy[0] / accuracy[1]

            print("Accuracy of the detection after epoch {}: {:.2f}".format(epoch, average_accuracy))
            

    callbacks = [checkpointer, tensorboard_callback, PlotOutput()]

    return callbacks



if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_dir", default="/data-ssd/dataset_fod/dataset_ok")
    parser.add_argument("--test_dir", default="/data-ssd/dataset_fod/dataset_test")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--log_dir", default="logs") 
    parser.add_argument("--weights_name", default="weights.h5")
    parser.add_argument("--num_parallel_calls", default=8)
    parser.add_argument("--n_patches", default=8)
    parser.add_argument("--patch_shape", default=(256, 256))
    parser.add_argument("--batch_size", default=10)
    parser.add_argument("--prefetch", default=100)
    parser.add_argument("--n_epochs", default=100)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--weight_decay", default=0.001)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--train_continue", default=False)
    parser.add_argument("--weights_path", default="models/2020-03-04-16-33-55/weights.h5")

    args = parser.parse_args()

    time_stamp = "{date:%Y-%m-%d-%H-%M-%S}".format(date=datetime.datetime.now())
    
    log_dir = os.path.join(args.log_dir, time_stamp)
    img_dir = log_dir + "-images"
    output_dir = os.path.join(args.output_dir, time_stamp)
    model_path = os.path.join(output_dir, args.weights_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)


    if args.optimizer == "adam":
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = SGD(lr=args.learning_rate, decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    else:
        raise ValueError("Unknown optimizer: {}".format(args.optimizer))


    # Define and save model topology to reconstruct it later
    num_filters = (16, 32, 64, 128, 256)
  
    with open(os.path.join(output_dir, "topology.txt"),"w") as topology:
        topology.write(str(num_filters))


    # Load data and initialize model
    files = glob.glob(os.path.join(args.file_dir, "*.tfrecord"))
    test_files = glob.glob(os.path.join(args.test_dir, "*.tfrecord"))

    random.shuffle(files)

    files = dataloader.split_files(files, train=0.8, valid=0.2)

    train_ds = dataloader.load_and_patch(files[0], "fit", args.patch_shape, args.n_patches, args.batch_size,
                         args.prefetch, args.num_parallel_calls, shuffle=None, repeat=True)
        
    valid_ds = dataloader.load_and_patch(files[1], "fit", args.patch_shape, args.n_patches, args.batch_size,
                         args.prefetch, args.num_parallel_calls, shuffle=None, repeat=True)

    test_ds, test_gt = dataloader.load_and_patch(test_files, "inf", num_parallel_calls=args.num_parallel_calls, batch_size=8)

    
    input_shape = (None, None, 3)
    
    model = Autoencoder(input_shape=input_shape, num_filters=num_filters)
    model = model.build()

    print(model.summary())

    if args.train_continue:
        model.load_weights(args.weights_path)


    # Train the model
    model.compile(optimizer=optimizer, loss="MSE", metrics=['accuracy'])
    history = model.fit(train_ds,
              steps_per_epoch=500,
              epochs=args.n_epochs,
              validation_data=valid_ds,
              validation_steps=250,
              callbacks=callbacks(model_path, test_ds, test_gt),
              verbose=1)

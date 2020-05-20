import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2



feature_description = {
    "input": tf.io.FixedLenFeature([2056, 2464, 3], tf.float32),
    "mask": tf.io.FixedLenFeature([2056, 2464, 1], tf.int64),          
    }


def parse_example(example):
    '''parse the serialized tf.Example using feature_description
    '''
    return tf.io.parse_single_example(example, feature_description)



def split_files(files, train=0.0, valid=0.0, test=0.0):
    '''
    files: list of .tfrecord files
    train, valid, test: float, percentual part of the total number of files divided by 100
    '''
    assert train + valid + test <= 1, 'Sum of the trainig, testing and validation partages is greater than 1.'

    n = len(files)

    train_index = int(n * train)
    valid_index = int(train_index + n * valid)
    test_index = int(valid_index + n * test)

    train_files = files[ :train_index]
    valid_files = files[train_index:valid_index]
    test_files = files[valid_index:test_index]

    return [train_files, valid_files, test_files]



def patch_example(example, ds_mode, patch_shape, n_patches):
    
    image = example["input"]

    inputs = []


    for i in range(n_patches):
        x = np.random.randint(0, 2464 - patch_shape[1]) 
        y = np.random.randint(0, 2056 - patch_shape[0])

        img_patch = image[y:y+patch_shape[0], x:x+patch_shape[1]]
        inputs.append(img_patch)


    inputs = tf.stack(inputs)

    if ds_mode == 0:
        return (inputs, inputs)
    
    if ds_mode == 1:
        return inputs



def load_and_patch(files, load_mode, patch_shape=(512, 512), n_patches=5, batch_size=10,
                   prefetch=2, num_parallel_calls=4, shuffle=None, repeat=True):
    '''
    load dataset from .tfrecord files, parse exmaples, cut the scenes into
    smaller patches and prepare the structure for the corresponding method
    from keras framework according to load_mode
    '''
    modes = {"fit": 0, "pred": 1, "inf": 2}

    assert load_mode in modes.keys(), "Invalid dataset loading mode, choose either fit, pred or inf for load_mode."

    ds_mode = modes[load_mode]

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_example, num_parallel_calls)

    if ds_mode == 2:
        dataset_in = dataset.map(lambda x: x["input"][8:,416:,:], num_parallel_calls)
        dataset_out = dataset.map(lambda x: x["mask"][8:,416:,:], num_parallel_calls)

        dataset_in = dataset_in.batch(batch_size)
        dataset_out = dataset_out.batch(batch_size)

        return dataset_in, dataset_out
    else:
        dataset = dataset.map(lambda x: patch_example(x, ds_mode, patch_shape, n_patches), num_parallel_calls)
        dataset = dataset.unbatch()
    
    if shuffle is not None:
        dataset = dataset.shuffle(shuffle)

    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(prefetch)

    return dataset



if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    files = glob.glob("dataset/*.tfrecord")
    dataset = load_and_patch(files, "inf")

    for inp, gt in dataset:
        img = inp[0].numpy()
        mask = gt[0].numpy()
        
        f = plt.figure()
        f.add_subplot(1, 2 , 1)
        plt.imshow(np.dstack((img, img, img)))
        f.add_subplot(1, 2 , 2)
        plt.imshow(np.dstack((mask, mask, mask)))
        plt.show(block=True)
        
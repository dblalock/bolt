#!/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import PIL
import pickle
import psutil  # pip install psutil
import shutil
import sys  # just for stderr for warnings
# import warnings

from PIL import Image

from python import files
from python import image_utils


from joblib import Memory
_memory = Memory('.', verbose=1)


IMAGENET_ONE_OF_EACH_PATH = '../datasets/one-of-each-imagenet'
IMAGENET_ONE_OF_EACH_FLOW_PATH = '../datasets/one-of-each-imagenet-as-folders'
# IMAGENET_64_PATH = os.path.expanduser("~/Desktop/datasets/imagenet64")
# IMAGENET_TINY_PATH = os.path.expanduser("~/Desktop/datasets/tiny-imagenet-200")
IMAGENET_64_PATH = '../datasets/imagenet64'
IMAGENET_TINY_PATH = '../datasets/tiny-imagenet-200'
IMAGENET_64_TRAIN_CHUNK_NSAMPLES = 128116

IMAGENET_TRAIN_PATH = '../datasets/ILSVRC2012/ILSVRC2012_img_train'
IMAGENET_TEST_PATH = '/home/dblalock/datasets/ILSVRC2012/ILSVRC2012_img_val'
if not os.path.exists(IMAGENET_TEST_PATH):  # try to load local version
    IMAGENET_TEST_PATH = '../datasets/ILSVRC2012/ILSVRC2012_img_val'

IMAGENET_10_CLASSES_TRAIN_PATH = '../datasets/ILSVRC2012_10/ILSVRC2012_img_train'
IMAGENET_10_CLASSES_TEST_PATH = '../datasets/ILSVRC2012_10/ILSVRC2012_img_val'
IMAGENET_100_CLASSES_TRAIN_PATH = '../datasets/ILSVRC2012_100/ILSVRC2012_img_train'
IMAGENET_100_CLASSES_TEST_PATH = '../datasets/ILSVRC2012_100/ILSVRC2012_img_val'

IMAGENET_1_EXAMPLE_TRAIN_PATH = '../datasets/imagenet-001-of-each'
IMAGENET_10_EXAMPLES_TRAIN_PATH = '../datasets/imagenet-010-of-each'
IMAGENET_25_EXAMPLES_TRAIN_PATH = '../datasets/imagenet-025-of-each'
IMAGENET_50_EXAMPLES_TRAIN_PATH = '../datasets/imagenet-050-of-each'
IMAGENET_100_EXAMPLES_TRAIN_PATH = '../datasets/imagenet-100-of-each'


# ================================================================ Downsampled

def _unpickle_file(path):
    with open(path, 'rb') as f:
        pydict = pickle.load(f)
    return pydict


# @_memory.cache
def _load_downsampled_data_file(path, layout='nhwc', dtype=None,
                                X_out=None, y_out=None, start_row=None):
    d = _unpickle_file(path)
    X = d['data']
    # NOTE: subtracting 1 so min idx is 0; this breaks synset lookup
    y = np.array(d['labels'], dtype=np.int32) - 1
    y = y.ravel()  # shouldn't be necessary, but might as well
    assert X.shape[0] == y.shape[0]
    assert len(X.shape) == 2

    nchan = 3
    npixels = X.shape[1] / nchan
    assert npixels * nchan == X.shape[1]  # each row not one img?
    side_len = int(np.sqrt(npixels))
    assert side_len * side_len == npixels
    X = X.reshape(X.shape[0], nchan, side_len, side_len)

    layout = 'nhwc' if layout is None else layout
    assert layout in ('nhwc', 'nchw')
    if layout == 'nhwc':
        X = np.moveaxis(X, 1, -1)  # make channels last axis
    X = np.ascontiguousarray(X)

    if X_out is not None:
        assert dtype in (None, X_out.dtype)
        dtype = X_out.dtype

    if dtype is not None:
        X = X.astype(dtype)

    # print("X shape: ", X.shape)
    # print("y shape: ", y.shape)

    if start_row is not None:
        end_row = start_row + X.shape[0]
    if X_out is not None:
        assert start_row is not None
        X_out[start_row:end_row] = X
    if y_out is not None:
        assert start_row is not None
        y_out[start_row:end_row] = y

    return X, y


def load_train_file_64x64(idx, verbose=0, **kwargs):
    assert idx in np.arange(1, 11)  # valid indices are 1 thru 10
    path = os.path.join(IMAGENET_64_PATH, "train_data_batch_{}".format(idx))
    if verbose > 1:
        print("loading train file: ", path)
    return _load_downsampled_data_file(path, **kwargs)


def _clean_which_file_idxs(which_file_idxs=None, dtype=None):
    if which_file_idxs is None:
        which_file_idxs = np.arange(1, 11)
    which_file_idxs = np.asarray(which_file_idxs, dtype=np.int32)

    # don't try to load more training data then we can actually fit in RAM
    mem_available = psutil.virtual_memory().available
    itemsize = dtype.itemsize if dtype is not None else 1
    one_img_nbytes = 64 * 64 * 3 * itemsize
    one_file_nbytes = IMAGENET_64_TRAIN_CHUNK_NSAMPLES * one_img_nbytes
    max_nfiles = (mem_available // one_file_nbytes) - 1
    # print("one_img_nbytes", one_img_nbytes)
    # print("one_file_nbytes", one_file_nbytes)
    # print("available mem", mem_available)
    # print("max_nfiles", max_nfiles)
    if max_nfiles < 1:
        raise MemoryError(
            "Minimum amount of RAM needed to load one chunk of ImageNet64x64 "
            "is {}B, but only {}B are available".format(
                one_file_nbytes, mem_available))
    requested_nfiles = len(which_file_idxs)
    if max_nfiles < requested_nfiles:
        requested_nbytes = (requested_nfiles + 1) * one_file_nbytes
        requested_MB = requested_nbytes // int(1e6)
        available_MB = mem_available // int(1e6)
        print("imagenet.load_train_data_64x64: MemoryWarning: "
              "Only loading {}/10 chunks of ImageNet64 instead of requested "
              "{}/10 since not enough memory; would need {:}MB, but only {:}MB "
              "are available".format(
                max_nfiles, requested_nfiles, requested_MB, available_MB),
              file=sys.stderr)
        # warnings.warn(msg, UserWarning)
        which_file_idxs = which_file_idxs[:max_nfiles]

    assert np.min(which_file_idxs) >= 1
    assert np.max(which_file_idxs) <= 10

    return which_file_idxs


# NOTE: total size of training data is around 16GB
def load_train_data_64x64(which_file_idxs=None, layout='nhwc', dtype=None,
                          verbose=1):
    which_file_idxs = _clean_which_file_idxs(which_file_idxs, dtype=dtype)

    if verbose > 0:
        print("load_train_data_64x64: loading file numbers: ", which_file_idxs)
    # import sys; sys.exit()

    if dtype is None:
        dtype = np.uint8  # default dtype

    # preallocate output matrix of appropriate size so that we can just
    # keep one copy of the data in memory (as opposed to loading all the
    # data matrices and then concatenating them)
    assert layout in ('nhwc', 'nchw')
    nrows_per_file = IMAGENET_64_TRAIN_CHUNK_NSAMPLES
    img_shape = (64, 64, 3) if layout == 'nhwc' else (3, 64, 64)
    combined_nrows = nrows_per_file * len(which_file_idxs)
    combined_shape = (combined_nrows,) + img_shape
    X_combined = np.zeros(combined_shape, dtype=dtype)
    y_combined = np.zeros(combined_nrows, dtype=np.int32)

    for i, idx in enumerate(which_file_idxs):
        start_row = nrows_per_file * i
        load_train_file_64x64(
            idx, layout=layout, X_out=X_combined, y_out=y_combined,
            start_row=start_row, verbose=verbose)

    return X_combined, y_combined


def load_test_data_64x64(layout='nhwc', dtype=None):
    path = os.path.join(IMAGENET_64_PATH, "val_data")
    return _load_downsampled_data_file(path, layout=layout, dtype=dtype)


# ================================================================ Tiny

# # adapted from https://github.com/keras-team/keras-preprocessing/blob/master/
# # keras_preprocessing/image/utils.py under MIT license
# def img_to_array(img, layout='nhwc', dtype='float32', mode='RGB'):
#     """Converts a PIL Image instance to a Numpy array.
#     # Arguments
#         img: PIL Image instance.
#         layout: Image data format, either "nchw" or "nhwc".
#         dtype: Dtype to use for the returned array.
#     # Returns
#         A 3D Numpy array.
#     # Raises
#         ValueError: if invalid `img` or `layout` is passed.
#     """
#     # print("img info:", img.format, img.size, img.mode)

#     # if img.mode == 'L':
#     if img.mode != mode:
#         img = img.convert(mode=mode)

#     if layout not in ('nchw', 'nhwc'):
#         raise ValueError('Unknown layout: %s' % layout)
#     # Numpy array x has format (height, width, channel)
#     # or (channel, height, width)
#     # but original PIL image has format (width, height, channel)
#     x = np.asarray(img, dtype=dtype)
#     if len(x.shape) == 3:
#         if layout == 'nchw':
#             x = x.transpose(2, 0, 1)
#     elif len(x.shape) == 2:
#         # print("x is only rank 2...WTF!?")
#         if layout == 'nchw':
#             x = x.reshape((1, x.shape[0], x.shape[1]))
#         else:
#             x = x.reshape((x.shape[0], x.shape[1], 1))
#     else:
#         raise ValueError('Unsupported image shape: %s' % (x.shape,))
#     return x


# def _resize_img(img, ratio_or_size):
#     if ratio_or_size is None or np.min(ratio_or_size) < 0:
#         return img
#     try:
#         nrows = ratio_or_size[0]
#         ncols = ratio_or_size[1]
#     except AttributeError:
#         nrows = img.height * ratio_or_size
#         ncols = img.width * ratio_or_size
#     new_size = (nrows, ncols)

#     is_downsampling = (nrows < img.height) or (ncols < img.width)
#     interp = PIL.Image.LANCZOS if is_downsampling else PIL.Image.BICUBIC
#     return img.resize(new_size, resample=interp)


# def image_utils.load_jpg(path, layout='nhwc', dtype='float32', resample=None):
#     img = Image.open(path)
#     img = _resize_img(img, ratio_or_size=resamp)
#     return img_to_array(img, layout=layout, dtype=dtype)


@_memory.cache
def _load_tiny_clsids_to_nums():
    wnids_path = os.path.join(IMAGENET_TINY_PATH, 'wnids.txt')
    with open(wnids_path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return {s: i for i, s in enumerate(lines)}


def _imagenet_tiny_cls_to_number(classnames):
    if isinstance(classnames, str):
        return _load_tiny_clsids_to_nums()[classnames]
    return [_load_tiny_clsids_to_nums()[name] for name in classnames]


@_memory.cache
def load_train_data_tiny(layout='nhwc', dtype=None, verbose=1):
    train_dir = os.path.join(IMAGENET_TINY_PATH, 'train')
    subdirs = files.list_subdirs(train_dir)
    all_classes = subdirs
    assert len(all_classes) == 200  # wrong number of classes??
    subdir_paths = files.list_subdirs(train_dir, abs_paths=True)

    all_imgs = []
    all_labels = []
    for i, pth in enumerate(np.sort(subdir_paths)):
        classname = os.path.basename(pth)
        if verbose > 0:
            print("loading images for class {}...".format(classname))

        imgs_subdir = os.path.join(pth, 'images')
        img_paths = files.list_files(
            imgs_subdir, endswith='.JPEG', abs_paths=True)
        assert len(img_paths) == 500  # supposed to be 500 examples per class...
        imgs = [image_utils.load_jpg(f, layout=layout,
                                     dtype=dtype)[np.newaxis, :, :, :]
                for f in img_paths]
        all_imgs += imgs
        lbl = _imagenet_tiny_cls_to_number(classname)
        all_labels += [lbl] * len(img_paths)

    X = np.concatenate(all_imgs, axis=0)
    y = np.array(all_labels, dtype=np.int32)

    return X, y


@_memory.cache
def load_test_data_tiny(layout='nhwc', dtype=None):
    # no labels given for "true" test set, so use the "val" subset as the
    # test set
    test_dir = os.path.join(IMAGENET_TINY_PATH, 'val')
    imgs_subdir = os.path.join(test_dir, 'images')
    img_paths = files.list_files(
        imgs_subdir, endswith='.JPEG', abs_paths=True)
    assert len(img_paths) == 10000  # wrong number of val images?

    # load images
    imgs = [image_utils.load_jpg(f, layout=layout,
                                 dtype=dtype)[np.newaxis, :, :, :]
            for f in img_paths]
    X = np.concatenate(imgs, axis=0)

    # load labels  # TODO make sure this computation is correct
    lbls_path = os.path.join(test_dir, 'val_annotations.txt')
    with open(lbls_path, 'r') as f:
        lines = f.readlines()
    fnames = [line.split()[0] for line in lines]
    class_ids = [line.split()[1] for line in lines]
    # complicated way that doesn't rely on annotations being sorted
    fname_to_class_id = dict(zip(fnames, class_ids))
    img_fnames = [os.path.basename(pth) for pth in img_paths]
    img_class_ids = [fname_to_class_id[fname] for fname in img_fnames]
    labels = _imagenet_tiny_cls_to_number(img_class_ids)
    y = np.array(labels, dtype=np.int32)

    return X, y


# ================================================================ K-of-each

# def load_data_one_of_each(layout='nhwc', dtype=None, size=None):
def load_data_one_of_each(layout='nhwc', dtype=None, size=(224, 224)):
    # np_save_file = os.path.join(IMAGENET_ONE_OF_EACH_PATH, 'oneOfEach.npy')
    # cached_exists = os.path.exists(np_save_file)
    # if cached_exists:
    #     return np.load(np_save_file)

    img_paths = files.list_files(IMAGENET_ONE_OF_EACH_PATH, endswith='.JPEG',
                                 abs_paths=True)
    assert len(img_paths) == 1000   # should be 1000 images...
    imgs = [image_utils.load_jpg(f, layout=layout, dtype=dtype, resample=size)
            for f in img_paths]

    if size is not None:  # can only concat if same size
        imgs = [img[np.newaxis, :, :, :] for img in imgs]
        X = np.concatenate(imgs, axis=0)
    else:
        X = imgs

    # XXX this is a total hack that will break if we get >1 img per class, and
    # already (probably) doesn't match up with the synsets
    # lbls = [os.path.basename(path).split('_')[0] for path in img_paths]
    y = np.arange(len(X))
    return X, y


# ================================================================ example

def load_flow_example(**kwargs):
    IMAGENET_EXAMPLE_PATH = os.path.abspath('../datasets/imagenet-example')

    # print("files in data dir:")
    # print(files.list_subdirs(IMAGENET_EXAMPLE_PATH))
    # import sys; sys.exit()

    j = os.path.join
    kwargs.setdefault('target_size', (224, 224))
    kwargs.setdefault('batch_size', 16)
    kwargs.setdefault('class_mode', 'categorical')

    import keras
    from keras.preprocessing import image
    train_datagen = image.ImageDataGenerator()
    val_datagen = image.ImageDataGenerator()
    test_datagen = image.ImageDataGenerator()

    # print("contents of train dir: ", files.list_subdirs(j(IMAGENET_EXAMPLE_PATH, 'train')))
    train_generator = train_datagen.flow_from_directory(
        j(IMAGENET_EXAMPLE_PATH, 'train'),
        **kwargs)
    val_generator = val_datagen.flow_from_directory(
        j(IMAGENET_EXAMPLE_PATH, 'val'),
        **kwargs)
    test_generator = val_datagen.flow_from_directory(
        j(IMAGENET_EXAMPLE_PATH, 'val'),
        **kwargs)

    return train_generator, val_generator, test_generator


def example_imagenet_train():
    import tensorflow as tf
    import keras
    from python import models
    from python import approx_conv_v2 as aconv

    # both are necessary to actually get consistent output
    np.random.seed(123)
    tf.random.set_random_seed(123)

    model = models.get_model(models.VGG16, weights=None, input_shape=(224, 224, 3))

    # swap out normal conv layer with our custom layer
    model = models.replace_layer_classes(
        model, {keras.layers.Conv2D: aconv.MyConv2D})

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    train_generator, val_generator, test_generator = load_flow_example()
    model.fit_generator(train_generator, steps_per_epoch=10, epochs=1,
                        validation_steps=1, validation_data=val_generator)

    model.evaluate_generator(test_generator, steps=2)


def main():
    # X, y = load_train_file_64x64(1)
    # X, y = load_train_data_64x64([1, 2, 3])
    # X, y = load_train_data_64x64([1, 2, 3, 4, 5])  # works
    # X, y = load_train_data_64x64()  # correctly yields mem warning
    # X, y = load_train_data_64x64([1])
    # X, y = load_test_data_64x64()
    # X, y = load_data_one_of_each(size=None)  # no resampling
    # X, y = load_data_one_of_each(size=(224, 224))

    # wow, imagenet-tiny looks like crap; lots of aliasing
    X, y = load_train_data_tiny()
    # X, y = load_test_data_tiny()

    print("X, y dtypes and shapes:")
    print(X.dtype)
    print(y.dtype)
    print(X.shape)
    print(y.shape)

    import matplotlib.pyplot as plt
    inp = 'y'
    count = 0
    while inp == 'y':
        _, axes = plt.subplots(3, 3, figsize=(9, 9))
        idxs = np.random.choice(np.arange(len(X)), size=axes.size)
        # offset = 0
        # offset = 10000
        # idxs = np.arange(offset + 9*count, offset + 9 + 9*count)
        for i, ax in enumerate(axes.ravel()):
            idx = idxs[i]
            img, classId = X[idx], y[idx]
            ax.imshow(img, interpolation='nearest')
            ax.set_title("Idx = {}, class = {}".format(idx, classId))
        # plt.imshow(X[100*1000], interpolation='nearest')
        # plt.imshow(X[300*1000], interpolation='nearest')
        plt.tight_layout()
        plt.show()

        count += 1
        inp = input("enter y to plot more random imgs; anything else to stop: ")


def _folderize_imagenet_one_of_each():  # one-off script
    olddir = IMAGENET_ONE_OF_EACH_PATH
    newdir = IMAGENET_ONE_OF_EACH_FLOW_PATH
    files.ensure_dir_exists(newdir)
    old_files = files.list_files(olddir, endswith='.JPEG', abs_paths=True)
    for f in files.list_files(olddir, endswith='.JPEG', abs_paths=True):
        basename = os.path.basename(f)
        label = basename.split('_')[0]
        subdir = os.path.join(newdir, label)
        files.ensure_dir_exists(subdir)
        newpath = os.path.join(subdir, basename)
        # newpath = os.path.join(newdir, )
        # print("oldpath: ", f, os.path.exists(f))
        # print("newpath: ", newpath)
        shutil.copy(f, newpath)


def _make_imagenet_k_of_each(k=10):
    out_path = '../datasets/imagenet-{:03d}-of-each'.format(k)
    print("writing to path: ", out_path)
    src_dir = IMAGENET_TRAIN_PATH
    for synset in files.list_subdirs(src_dir):
        subdir_path = os.path.join(src_dir, synset)
        img_paths = sorted(files.list_files(subdir_path, abs_paths=True))
        img_paths = img_paths[:k]

        new_subdir = os.path.join(out_path, synset)
        files.ensure_dir_exists(new_subdir)
        for path in img_paths:
            fname = os.path.basename(path)
            new_path = os.path.join(new_subdir, fname)
            shutil.copy(path, new_path)


if __name__ == '__main__':
    # example_imagenet_train()
    main()
    # _folderize_imagenet_one_of_each()
    # _make_imagenet_k_of_each(10)
    # _make_imagenet_k_of_each(25)
    # _make_imagenet_k_of_each(50)
    # _make_imagenet_k_of_each(100)


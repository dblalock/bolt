#!/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import os

import PIL
from PIL import Image
from PIL import ImageOps  # can't just do PIL.ImageOps for some reason

from . import files


# ================================ TODO rm duplicate code from imagenet.py

# adapted from https://github.com/keras-team/keras-preprocessing/blob/master/
# keras_preprocessing/image/utils.py under MIT license
def img_to_array(img, layout='nhwc', dtype='float32', mode='RGB'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        layout: Image data format, either "nchw" or "nhwc".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `layout` is passed.
    """
    # print("img info:", img.format, img.size, img.mode)

    # if img.mode == 'L':
    if img.mode != mode:
        img = img.convert(mode=mode)

    if layout not in ('nchw', 'nhwc'):
        raise ValueError('Unknown layout: %s' % layout)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)

    if len(x.shape) == 3:
        if layout == 'nchw':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        # print("x is only rank 2...WTF!?")
        if layout == 'nchw':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def resize_img(img, ratio_or_size):
    if ratio_or_size is None or np.max(ratio_or_size) < 0:
        return img
    try:
        nrows = ratio_or_size[0]
        ncols = ratio_or_size[1]
        nrows = img.height if nrows < 0 else nrows
        ncols = img.width if ncols < 0 else ncols
    except AttributeError:
        nrows = img.height * ratio_or_size
        ncols = img.width * ratio_or_size
    new_size = (nrows, ncols)

    is_downsampling = (nrows < img.height) or (ncols < img.width)
    interp = PIL.Image.LANCZOS if is_downsampling else PIL.Image.BICUBIC
    return img.resize(new_size, resample=interp)


def crop_img(img, crop_how=None, new_size=(224, 224), resize_shorter_to=256):
    if crop_how is None:
        return img
    assert crop_how in ('center', 'square')

    height, width = img.height, img.width
    if (height == width) and (new_size is None):
        return img

    if crop_how == 'center':
        return center_crop(img, new_size=new_size,
                           resize_shorter_to=resize_shorter_to)

    if new_size is None:
        new_width = min(width, height)
        new_height = new_width
    else:
        new_height, new_width = new_size

    assert new_width <= width
    assert new_height <= height

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    # right = (width + new_width) // 2
    # bottom = (height + new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return img.crop((left, top, right, bottom))


def center_crop(img, new_size=(224, 224), resize_shorter_to=256):
    height, width = img.height, img.width
    minsize = min(height, width)
    new_height = (height * resize_shorter_to) // minsize
    new_width = (width * resize_shorter_to) // minsize
    img = img.resize(
        (new_width, new_height), resample=Image.BICUBIC)

    assert min(new_width, new_height) == resize_shorter_to

    return crop_img(img, crop_how='square', new_size=new_size)


def pad_img(img, pad_how='square', fill_value=0):
    if pad_how is None:
        return img
    assert pad_how == 'square'  # no other kinds of cropping supported

    height, width = img.height, img.width
    if height == width:
        return img

    new_size = max(height, width)
    delta_w = new_size - width
    pad_left = delta_w // 2
    pad_right = delta_w - pad_left
    delta_h = new_size - height
    pad_top = delta_h // 2
    pad_bottom = delta_h - pad_top

    padding = pad_left, pad_top, pad_right, pad_bottom
    return ImageOps.expand(img, border=padding, fill=fill_value)


def load_jpg(path, layout='nhwc', dtype=None, resample=None,
             crop=None, pad=None):
    img = PIL.Image.open(path)
    img = pad_img(img, pad)
    img = crop_img(img, crop)
    img = resize_img(img, ratio_or_size=resample)
    return img_to_array(img, layout=layout, dtype=dtype)


# assumes one subdir for each class, with class name equal to subdir name
# @_memory.cache
def load_jpegs_from_dir(dirpath, remove_classes=None, require_suffix=None,
                        layout='nhwc', dtype=None, resample=(224, 224),
                        crop='center', pad=None, verbose=1,
                        limit_per_class=None, only_return_path=False):
    subdirs = sorted(files.list_subdirs(dirpath, only_visible=True))
    if remove_classes is not None:
        if isinstance(remove_classes, str):
            remove_classes = [remove_classes]
        for classname in remove_classes:
            subdirs.remove(classname)

    if verbose > 0:
        print("found {} classes in directory: {}".format(len(subdirs), dirpath))

    classname_to_label = {name: i for i, name in enumerate(subdirs)}  # noqa
    label_to_classname = {i: name for name, i in classname_to_label.items()}

    all_imgs = []
    all_labels = []
    for subdir in subdirs:
        subdir_path = os.path.join(dirpath, subdir)
        img_paths = files.list_files(
            subdir_path, endswith=require_suffix, abs_paths=True,
            only_visible=True)

        if limit_per_class is not None and limit_per_class > 0:
            img_paths = img_paths[:limit_per_class]

        if verbose > 1:
            print("loading {:4d} images for class '{}'".format(
                len(img_paths), subdir))

        # not certain += [...] version was working
        label = classname_to_label[subdir]
        for i in range(len(img_paths)):
            all_labels.append(label)
        # all_labels += [] * len(img_paths)
        if only_return_path:
            imgs = img_paths
        else:
            imgs = [load_jpg(f, layout=layout, dtype=dtype, resample=resample,
                             crop=crop, pad=pad)[np.newaxis, :, :, :]
                    for f in img_paths]
        all_imgs += imgs

    if only_return_path:
        X = all_imgs
    else:
        try:
            # this works iff resampled/padded/cropped to same size
            X = np.concatenate(all_imgs, axis=0)
        except ValueError:
            # otherwise strip batch dim (so each image is 3D)
            X = [img.reshape(img.shape[1:]) for img in all_imgs]
    y = np.array(all_labels, dtype=np.int32)
    return (X, y), label_to_classname

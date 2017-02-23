#!/bin/env/python

import os
import shutil


def ls():
    return os.listdir('.')


def is_hidden(path):
    return os.path.basename(path).startswith('.')


def is_visible(path):
    return not is_hidden(path)


def join_paths(dir, contents):
    return map(lambda f: os.path.join(dir, f), contents)


def files_matching(dir, prefix=None, suffix=None, absPaths=False,
                   onlyFiles=False, onlyDirs=False):
    files = os.listdir(dir)
    if prefix:
        files = filter(lambda f: f.startswith(prefix), files)
    if suffix:
        files = filter(lambda f: f.endswith(suffix), files)
    if onlyFiles or onlyDirs:
        paths = join_paths(dir, files)
        if onlyFiles:
            newFiles = []
            for f, path in zip(files, paths):
                if os.path.isfile(path):
                    newFiles.append(f)
            files = newFiles
        if onlyDirs:
            newFiles = []
            for f, path in zip(files, paths):
                if os.path.isdir(path):
                    newFiles.append(f)
            files = newFiles
    if absPaths:
        files = join_paths(dir, files)
    return files


def list_subdirs(dir, startswith=None, endswith=None, absPaths=False):
    return files_matching(dir, startswith, endswith, absPaths,
                          onlyDirs=True)


def list_files(dir, startswith=None, endswith=None, absPaths=False):
    return files_matching(dir, startswith, endswith, absPaths,
                          onlyFiles=True)


def list_hidden_files(dir, startswith=None, endswith=None, absPaths=False):
    contents = files_matching(dir, startswith, endswith, absPaths,
                              onlyFiles=True)
    return filter(is_hidden, contents)


def list_visible_files(dir, startswith=None, endswith=None, absPaths=False):
    contents = files_matching(dir, startswith, endswith, absPaths,
                              onlyFiles=True)
    return filter(is_visible, contents)


def remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except (OSError):
            shutil.rmtree(path)


def force_create_dir(dir):
    if os.path.exists(dir):
        remove(dir)
    os.makedirs(dir)


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def basename(f, noext=False):
    name = os.path.basename(f)
    if noext:
        name = name.split('.')[0]
    return name

#!/bin/env/python

import os
import shutil


def ls(dir='.'):
    return os.listdir(dir)


def is_hidden(path):
    return os.path.basename(path).startswith('.')


def is_visible(path):
    return not is_hidden(path)


def join_paths(dir, contents):
    return [os.path.join(dir, f) for f in contents]


def files_matching(dir, prefix=None, suffix=None, abs_paths=False,
                   only_files=False, only_dirs=False, recursive=False,
                   only_visible=False, only_hidden=False):
    files = os.listdir(dir)
    if recursive:
        abs_dir = dir
        paths = join_paths(abs_dir, files)
        for path in paths:
            if not os.path.isdir(path):
                continue
            matches = files_matching(
                path, prefix=prefix, suffix=suffix,
                abs_paths=abs_paths, only_files=only_files,
                only_dirs=only_dirs, recursive=True)
            matches = join_paths(path, matches)
            matches = [os.path.relpath(m, start=dir) for m in matches]
            files += matches

    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    if suffix:
        files = [f for f in files if f.endswith(suffix)]
    if only_files or only_dirs:
        paths = join_paths(dir, files)
        if only_files:
            files = [f for f, p in zip(files, paths) if os.path.isfile(p)]
        if only_dirs:
            files = [f for f, p in zip(files, paths) if os.path.isdir(p)]
    if abs_paths:
        files = join_paths(os.path.abspath(dir), files)
    if only_visible:
        files = [f for f in files if is_visible(f)]
    if only_hidden:
        files = [f for f in files if is_hidden(f)]

    return sorted(files)


def list_subdirs(dir, startswith=None, endswith=None, abs_paths=False,
                 recursive=False):
    return files_matching(dir, startswith, endswith, abs_paths,
                          only_dirs=True, recursive=recursive)


def list_files(dir, startswith=None, endswith=None, abs_paths=False,
               recursive=False):
    return files_matching(dir, startswith, endswith, abs_paths,
                          only_files=True, recursive=recursive)


def list_hidden_files(dir, startswith=None, endswith=None, abs_paths=False,
                      recursive=False):
    return files_matching(dir, startswith, endswith, abs_paths, only_files=True,
                          recursive=recursive, only_hidden=True)


def list_visible_files(dir, startswith=None, endswith=None, abs_paths=False,
                       recursive=False):
    return files_matching(dir, startswith, endswith, abs_paths, only_files=True,
                          recursive=recursive, only_visible=True)


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


def ensure_dir_exists(dir_or_file):
    if '.' in os.path.basename(dir_or_file):  # this looks like a file
        dirname = os.path.dirname(dir_or_file)
    else:
        dirname = dir_or_file
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def basename(f, noext=False):
    name = os.path.basename(f)
    if noext:
        name = name.split('.')[0]
    return name

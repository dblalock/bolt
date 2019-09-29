#!/bin/env/python

"""utility functions for running experiments"""

from __future__ import print_function, absolute_import

import datetime
import os
import itertools
import warnings
import numpy as np
import pandas as pd
import sys

import sklearn
# from sklearn.model_selection import StratifiedKFold

from python.files import ensure_dir_exists

try:
    from joblib import Memory
    memory = Memory('.', verbose=0)
    cache = memory.cache
except Exception:
    def cache(f):
        return f

# ================================================================ Constants

KEY_FINISHED_UPDATING = '__pyn_finished_updating__'
KEY_NEW_KEYS = '__pyn_newkeys__'


# ================================================================ Types

class UsageError(Exception):
    pass


class Options(object):
    """Wrapper for a collection to signify that each element is one possible
    parameter value"""

    def __init__(self, *args):
        if args is None or len(args) < 1:
            raise ValueError("No options given!")

        if len(args) == 1 and hasattr(args, '__len__'):
            self.values = args[0]  # given a list
        else:
            self.values = args  # given individual objects

    def __len__(self):
        return len(self.values)

    # deliberately don't act like a collection so that we fail fast if
    # code doesn't know that this is supposed to represent Options, rather
    # than a collection of values. This is mostly to ensure that Options
    # are always expanded out when generating sets of parameters.
    def __getitem__(self, idx):
        self._raise()

    def __setitem__(self, idx, item):
        self._raise()

    def _raise(self):
        raise TypeError("Options object is not a collection; use options.values"
                        " to access the collection of individual options")


# ================================================================ Funcs

# ------------------------------------------------ misc utils

def make_immutable(x):
    """
    >>> make_immutable(5) == 5
    True
    >>> make_immutable('a') == 'a'
    True
    >>> make_immutable((1, 2)) == (1, 2)
    True
    >>> make_immutable([1, 2]) == [1, 2]
    False
    >>> make_immutable([1, 2]) == (1, 2)
    True
    """
    # must either be not a collections or immutable
    try:
        {}[x] = 0   # dicts require immutability
        return x
    except TypeError:
        # so it's mutable; either a collection or a
        # mutable class; if a class, we're hosed, so
        # assume it's a collection
        try:
            # if it's a singleton collection, try returning
            # first element; this will jump to except
            # unless x is a collection
            _ = len(x)

            # apparently a collection, so make it a tuple
            return tuple(x)
        except TypeError:
            return repr(x)  # not a collection; stringify as last resort


def as_key(x):
    return make_immutable(x)


# ------------------------------------------------ IO / saving results

def now_as_string():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")


def save_data_frame(df, save_dir='results', name="", timestamp='copy',
                    cols_in_filename=None, col_kv_fmt="_{}={}",
                    store_index=False, append=True, dedup_cols=None,
                    add_timestamp_col=True, sort_by=None, **sink):
    if timestamp == 'copy':  # save one copy with and without timestamp
        kwargs = dict(name=name, col_kv_fmt=col_kv_fmt,
                      cols_in_filename=cols_in_filename, dedup_cols=dedup_cols,
                      store_index=store_index, append=append, sort_by=sort_by,
                      add_timestamp_col=add_timestamp_col)
        backups_dir = os.path.join(save_dir, 'pyience-backups')
        save_data_frame(df, timestamp=True, save_dir=backups_dir, **kwargs)
        save_data_frame(df, timestamp=False, save_dir=save_dir, **kwargs)
        return

    # construct filename
    name = name if name else ""
    if cols_in_filename:
        cols = list(df.columns.values)
        # substrs = ["{%s}" % col for col in cols]
        # name = name_fmt
        # for ss in substrs:
        #     key = ss[1:-1]
        for key in cols_in_filename:
            if key not in cols:
                warnings.warn("Column '{}' not found in Dataframe."
                              "Excluding it from filename".format(key))
                continue

            # get value associated with this key; ignored if col not constant
            vals = df[key]
            nuniq = len(vals.unique())
            if nuniq != 1:
                warnings.warn("Column '{}' has more than one value in Dataframe."
                              "Excluding it from filename".format(key))
                continue
            val = vals[0]

            fmt = col_kv_fmt
            if name == "" and not col_kv_fmt.startswith("{"):
                fmt = col_kv_fmt[1:]
            name += fmt.format(key, val)

    ensure_dir_exists(save_dir)
    raw_timestamp_str = now_as_string()
    timestamp_str = ("_" + raw_timestamp_str) if timestamp else ""
    fileName = "{}{}.csv".format(name, timestamp_str).strip("_")
    save_path = os.path.join(save_dir, fileName)

    if add_timestamp_col:
        df['__pyience_timestamp__'] = [raw_timestamp_str] * df.shape[0]

    if append and os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        # print("existing_df_cols", existing_df.columns)
        # print("existing_df_cols", df.columns)
        # print("dedup_cols", dedup_cols)
        df = pd.concat([existing_df, df], axis=0, sort=False, ignore_index=True)
        # print("df secs: ")
        # print(df['secs'])
        dedup_cols = set(dedup_cols) & set(list(df.columns))
        df.drop_duplicates(subset=dedup_cols, keep='last', inplace=True)

    df = df.sort_index(axis=1)
    if sort_by is not None:
        df.sort_values(sort_by, inplace=True)
        # also move these cols to the front for legibility, since they're
        # probably something you care about
        other_cols = [col for col in df.columns.values if col not in sort_by]
        df = df[sort_by + other_cols]

    df.to_csv(save_path, index=store_index)


def save_dicts_as_data_frame(d, **kwargs):
    if not isinstance(d, dict):
        try:
            df = pd.DataFrame.from_records(d)
        except Exception:
            dfs = [pd.DataFrame.from_records(dd, index=[0]) for dd in d]
            df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        df = pd.DataFrame.from_records(d, index=[0])
    save_data_frame(df, **kwargs)


def generate_save_path(params, savedir, subdir_keys=None):
    subdir = ''

    # create nested subdirectories with names specified by
    # the values for the keys in subdir_keys
    if subdir_keys is not None:
        subdir_keys = list(subdir_keys)

        subdir_names = ["{}__{}".format(str(key), str(params[key]))
                        for key in subdir_keys]

        subdir = os.path.join(*subdir_names)
        savedir = os.path.join(savedir, subdir)

    return savedir


# ------------------------------------------------ parameter generation

def expand_params(params):
    """dict of kv pairs -> list of dicts with one option selected for
    each key whose value is an instance of Options."""

    # keys with values that are Options; try all combos of these
    options_keys = [key for key in params if isinstance(params[key], Options)]
    options_keys = sorted(options_keys)  # sort for reproducibility
    options_vals = [params[key].values for key in options_keys]

    # keys with values that aren't Options; these are the same every time
    no_options_keys = [key for key in params if not isinstance(params[key], Options)]
    no_options_vals = [params[key] for key in no_options_keys]
    no_options_params = dict(zip(no_options_keys, no_options_vals))

    # make a list of all possible combos of values for each key with Options
    expanded_params_list = []
    for v in itertools.product(*options_vals):
        expanded_params = dict(zip(options_keys, v))  # pick one option for each
        expanded_params.update(no_options_params)  # add in fixed params
        expanded_params_list.append(expanded_params)

    return expanded_params_list


def update_func_from_dict(d):
    def f(params, new_keys, d=d):
        updated = False
        for k, v in d.items():
            if k in new_keys:
                for kk, vv in v.items():
                    updated = updated or (kk not in params)
                    params.setdefault(kk, vv)
        return updated
    return f


def generate_params_combinations(params_list, update_func={}):
    """Uses update_func to update each dict based on its values (e.g., to
    add SVM kernel params if it contains "classifier": "SVM")"""
    if not isinstance(params_list, (list, set, frozenset, tuple)):
        params_list = [params_list]

    for params in params_list:
        params[KEY_NEW_KEYS] = set(params.keys())

    if isinstance(update_func, dict):
        update_func = update_func_from_dict(update_func)

    while True:
        new_list = []
        for params in params_list:
            expanded = expand_params(params)
            new_list += expanded

        if not update_func:
            params_list = new_list
            break

        allFinished = True
        for params in new_list:
            # if these params aren't fully updated, update them; keep
            # track of which keys are added along the way so we can
            # pass this set to the update function next time
            if not params.get(KEY_FINISHED_UPDATING, False):
                # read which keys were added last time and which keys
                # are currently present
                new_keys = params[KEY_NEW_KEYS]
                existing_keys = frozenset(params.keys())
                params.pop(KEY_NEW_KEYS)

                unfinished = update_func(params, new_keys)

                # compute and store which keys were added this time
                new_keys = frozenset(params.keys()) - existing_keys
                params[KEY_NEW_KEYS] = new_keys

                if unfinished:
                    allFinished = False
                params[KEY_FINISHED_UPDATING] = not unfinished

        params_list = new_list

        if allFinished:
            break

    for p in params_list:
        p.pop(KEY_FINISHED_UPDATING)
        p.pop(KEY_NEW_KEYS)

    return params_list


# ------------------------------------------------ cross validation

def stratified_split_train_test(X, Y, train_frac=.8, random_state=123):
    """Returns X_train, X_test, y_train, y_test"""
    return sklearn.model_selection.train_test_split(
        X, Y, train_size=train_frac, stratify=Y, random_state=random_state)


def split_train_test(X, Y, train_frac=.8, random_state=123):
    """Returns X_train, X_test, y_train, y_test"""
    np.random.seed(123)
    return sklearn.model_selection.train_test_split(
        X, Y, train_size=train_frac, random_state=random_state)

    # n_folds = int(train_frac / (2. - train_frac))
    # split = StratifiedKFold(Y, n_folds=n_folds, random_state=12345)
    # train_index, test_index = next(iter(split))
    # X, Xtest = X[train_index], X[test_index]
    # Y, Ytest = Y[train_index], Y[test_index]
    # return X, Xtest, Y, Ytest


# ------------------------------------------------ Command line

def _split_kv_arg(arg):
    key, val = arg.split('=')
    return key.strip('-'), val


def _is_kv_arg(arg):
    return len(arg.split('=')) == 2


def _clean_flag_arg(arg):
    return arg.strip('-')


def _is_flag_arg(arg):
    return arg[0] == '-'


def _parse_func_call_cmd(s):
    """

    >>> _parse_func_call_cmd("range(5)")
    array([0, 1, 2, 3, 4])
    >>> _parse_func_call_cmd("range(2, -3, -2)")
    array([ 2,  0, -2])
    >>> _parse_func_call_cmd("linspace( -2,-20, 3)")
    array([ -2., -11., -20.])
    >>> _parse_func_call_cmd("logspace(-1, 3, 3)")
    array([1.e-01, 1.e+01, 1.e+03])
    """
    fnames = 'randn randint range linspace logspace'.split()
    nargs = [(1,), (1, 2, 3), (1, 2, 3), (2, 3), (2, 3)]
    funcs = [np.random.randn, np.random.randint, np.arange,
             np.linspace, np.logspace]

    if not isinstance(s, str):
        return None

    for fname, argc, func in zip(fnames, nargs, funcs):
        if not s.startswith(fname + '('):
            continue
        if not s.endswith(')'):
            raise ValueError("You tried to call function '{}', but forgot the"
                             " closing parenthesis".format(fname))
        in_parens = s[len(fname) + 1:-1]
        maybe_args = in_parens.split(',')
        if len(maybe_args) not in argc:
            raise ValueError(
                "You tried to call function '{}', but passed an invalid number"
                " of arguments: {}. Needed to be one of: {}" .format(
                    fname, len(maybe_args), argc))
        try:
            nums = [int(arg) for arg in maybe_args]
            return func(*nums)
        except:  # noqa
            raise ValueError("Command '{}' has arguments that can't be coerced"
                             " into integers".format(s))
    return None


def _to_appropriate_type(s):
    """convert string `s` to an int, bool, float, or integer range as
    appropriate. Returns the original string if it does not appear to be
    any of these types."""
    if s == 'True' or s == 'T':
        return True
    elif s == 'False' or s == 'F':
        return False
    try:
        return int(s)
    except:  # noqa
        pass
    try:
        return float(s)
    except:  # noqa
        pass
    if len(s.split('..')) in (2, 3):  # range
        vals_as_strs = s.split('..')
        try:
            return np.arange(*[int(val) for val in vals_as_strs])
        except:  # noqa
            pass
    as_func_result = _parse_func_call_cmd(s)
    if as_func_result is not None:
        return as_func_result
    return s


def parse_cmd_line(argv=None, positional_keys=None, allow_flags=True,
                   infer_types=True):
    """Parses the list of command line arguments into a dictionary of
    key-value pairs

    Parameters
    ----------
    argv : iterable of strings
        This should be sys.argv if supplied. Otherwise, sys.argv is read.

    positional_keys : iterable of strings, optional
        If k strings are specified, the up to the first k arguments will
        be treated as values to be paired with these keys. Arguments of the
        form foo=bar will never be treated this way.

    allow_flags : bool, optional
        If True, allows arguments of the form --myArg. When passed, this will
        add {'myArg': True} to the returned dictionary. This is equivalent to
        myArg=True

    infer_types : bool, optional
        If True, attempts to infer the type of each value in the returned
        dictionary. E.g., instead of returning {'height': '72'}, it will
        return {'height': 72}.

    Returns
    -------
    argKV : dict: string -> inferred type or string
        A dictionary whose keys and values are specified by the command line
        arguments

    >>> # ------------------------ positional args only
    >>> argv = ['pyience.py', 'fooVal', 'barVal']
    >>> d = parse_cmd_line(argv, positional_keys=['fooKey', 'barKey'])
    >>> len(d)
    2
    >>> d['fooKey']
    'fooVal'
    >>> d['barKey']
    'barVal'
    >>> # ------------------------ key-value args
    >>> argv = ['pyience.py', 'fooVal', 'bletchKey=bletchVal', 'blahKey=blahVal']
    >>> d = parse_cmd_line(argv, positional_keys=['fooKey', 'barKey'])
    >>> len(d)
    3
    >>> d['fooKey']
    'fooVal'
    >>> d.get('barKey', 'notHere')
    'notHere'
    >>> d['bletchKey']
    'bletchVal'
    >>> d['blahKey']
    'blahVal'
    >>> # ------------------------ flags
    >>> argv = ['pyience.py', 'fooVal', 'bletchKey=bletchVal', '--myFlag']
    >>> d = parse_cmd_line(argv, positional_keys=['fooKey', 'barKey'])
    >>> d['myFlag']
    True
    >>> # ------------------------ type inference
    >>> argv = ['pyience.py', '--myFlag', 'foo=1.1', 'bar=7', 'baz=T', 'r=1..5']
    >>> d = parse_cmd_line(argv, positional_keys=['fooKey', 'barKey'])
    >>> len(d)
    5
    >>> d['myFlag']
    True
    >>> d['foo']
    1.1
    >>> d['bar']
    7
    >>> d['baz']
    True
    >>> d['r']
    array([1, 2, 3, 4])
    >>> # ------------------------ no positional args
    >>> d = parse_cmd_line(argv)
    >>> len(d)
    5
    >>> d['myFlag']
    True
    >>> d['foo']
    1.1
    """

    if argv is None:
        argv = sys.argv

    args = argv[1:]  # ignore file name

    num_positional_keys = 0
    if positional_keys is not None and len(positional_keys):
        num_positional_keys = len(positional_keys)

    # validate input; keyword arguments must come after positional
    # arguments, and there must be no more positional arguments than
    # we have keys to associate with them
    kwargs_started = False
    flags_started = False
    for i, arg in enumerate(args):
        if _is_kv_arg(arg):  # it's a keyword argument
            kwargs_started = True
        elif _is_flag_arg(arg):
            flags_started = True
        else:  # it's not a keyword argument or flag arguemnt
            if kwargs_started:
                raise UsageError("key=value arguments must come after"
                                 "positional arguments!")

            if flags_started:
                raise UsageError("flag (e.g., --myFlag) arguments must come"
                                 "after positional arguments!")

            arg_num = i + 1
            if arg_num > num_positional_keys:
                raise UsageError("only expecting "
                                 "{} positional arguments!".format(
                                    num_positional_keys))

    argKV = {}
    for i, arg in enumerate(args):
        if _is_kv_arg(arg):
            key, val = _split_kv_arg(arg)
            argKV[key] = val
        elif _is_flag_arg(arg):
            key = _clean_flag_arg(arg)
            argKV[key] = 'True'  # string so that all vals are strings
        elif i < num_positional_keys:
            key = positional_keys[i]
            argKV[key] = arg
        else:
            raise UsageError("couldn't parse argument '{}'".format(arg))

    if infer_types:
        for k, v in argKV.items():
            argKV[k] = _to_appropriate_type(v)

    return argKV


# ------------------------------------------------ other stuff

def apply_funcs(funcs, data):
    f = chain(funcs)
    return f(data)


def chain(funcs):
    if funcs is None or not len(funcs):
        return lambda x: x

    def f(*args, **kwargs):
        res = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            res = func(res)

    return f


def subdict(d, keys):
    """Returns a new dictionary composed of the (key, value) pairs
    from d for the keys specified in keys"""
    return {k: d[k] for k in keys}


# ------------------------------------------------ sklearn interop

def set_attrs(obj, attrs_dict, require_attrs_exist=False):
    if require_attrs_exist:
        keys_and_there = ([(k, k in obj.__dict__) for k in attrs_dict])
        missing_keys = [k for (k, there) in keys_and_there if not there]
        there = zip(*keys_and_there)[1]
        if not all(there):
            raise ValueError("Object is missing keys {}".format(
                missing_keys))

    obj.__dict__.update(attrs_dict)


# ------------------------------------------------ cross validation

def _uniq_element_positions(iterable):
    """
    Returns a mapping of unique elements to positions at which they
    occur within the iterable
    """
    objs2positions = {}
    for i, obj in enumerate(iterable):
        key = as_key(obj)
        positions = objs2positions.get(key, [])
        positions.append(i)
        objs2positions[key] = positions
    return objs2positions


# def _group_start_idxs_eq_split(nelements, ngroups):
#   group_sz = nelements // ngroups
#   return np.arange(0, nelements, group_sz, dtype=np.int)

def _group_start_end_idxs(nelements, ngroups=-1, fractions=None):
    hasFracs = fractions is not None and len(fractions)

    if ngroups <= 1 and not hasFracs:
        return np.array([0], dtype=np.int), np.array([nelements], dtype=np.int)

    if not hasFracs:
        fracs = np.ones(ngroups)

    fractions = np.asarray(fracs)
    fractions /= np.max(fracs)

    cum_fracs = np.cumsum(fractions)
    end_idxs = (nelements * cum_fracs).astype(np.int)
    start_idxs = np.r_[0, end_idxs[:-1]]

    return start_idxs, end_idxs


def _split_into_groups(iterable, ngroups=-1, fractions=None, shuffle=True):
    if shuffle:
        iterable = np.copy(iterable)
        np.shuffle(iterable)

    start_idxs, end_idxs = _group_start_end_idxs(len(iterable), ngroups,
                                                 fractions)

    return [iterable[start:end] for start, end in zip(start_idxs, end_idxs)]


def cv_partition_idxs(labels, n_folds=5, fractions=None, stratified=True):
    if fractions is not None and len(fractions):
        if len(fractions) != n_folds:
            raise ValueError("Specified fractions of total for {} groups, but "
                             "n_folds is {}; ignoring n_fold".format(
                                len(fractions), n_folds))

    if stratified:
        all_idxs = [[] for i in range(n_folds)]
        lbl2idxs = _uniq_element_positions(labels)
        for lbl, idxs in lbl2idxs.items():
            if len(idxs) < n_folds:
                warnings.warn(("Label {} appears only {} times, which is "
                               "less than the number of folds requested, {}"
                               .format(lbl, len(idxs), n_folds)), Warning)

            idxGroups = _split_into_groups(idxs, n_folds, fractions)
            for i, group in enumerate(idxGroups):
                all_idxs[i] += group
        return all_idxs
    else:
        possible_idxs = np.arange(len(labels))
        return _split_into_groups(possible_idxs, n_folds, fractions)


def cv_split(X, y, n_folds=5, fractions=None, stratified=True):
    if len(X) != len(y):
        raise IndexError("len(X) {} != len(y) {}".format(len(X), len(y)))

    all_idxs = cv_partition_idxs(y, n_folds=n_folds, fractions=fractions,
                                 stratified=stratified)

    X_split = [X[idxs] for idxs in all_idxs]
    y_split = [y[idxs] for idxs in all_idxs]

    return X_split, y_split


# ================================================================ Main

def update(params, new_keys):
    if 'classifier' in new_keys:
        params['kernel'] = Options('rbf', 'linear')

    # we use setdefault here so that we don't overwrite values
    # passed in at the top level
    if 'kernel' in new_keys:
        kernel = params['kernel']
        params.setdefault('C', Options(10. ** np.arange(-5, 3)))
        if kernel == 'rbf':
            params.setdefault('gamma', Options([1, 10]))

    return True if new_keys else False


def main():
    cVals = 10. ** np.arange(-3, 3)
    d = {"classifier": "SVM", 'C': Options(cVals)}
    # generate_params_combinations(d, update)
    combos = generate_params_combinations(d, update)

    # add a fake outcome variable
    for combo in combos:
        combo['runtime'] = np.random.rand() * 10

    # print out a dataframe so we can see that this worked
    import pandas as pd
    print(pd.DataFrame.from_records(combos))  # woot; it worked


if __name__ == '__main__':
    from doctest import testmod
    testmod()

    main()

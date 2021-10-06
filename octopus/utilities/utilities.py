"""
Common utilities.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import shutil


def create_directory(path):
    if os.path.isdir(path):
        logging.info(f'Directory already exists:{path}.')
    else:
        os.mkdir(path)
        logging.info(f'Created directory:{path}.')


def delete_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        logging.info(f'Deleted directory:{path}.')
    else:
        logging.info(f'Directory does not exist:{path}.')


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)
        logging.info(f'Deleted file:{path}')
    else:
        logging.info(f'File does not exist:{path}')


def _to_int_list(s):
    return [int(a) for a in s.strip().split(',')]


def _to_string_list(s):
    return s.strip().split(',')


def _to_int_dict(s):
    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to an int
        try:
            val = int(val)
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d


def _to_float_dict(s):
    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to a float
        try:
            val = float(val)
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d
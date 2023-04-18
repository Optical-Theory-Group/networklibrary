# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:53:42 2022

@author: Matthew Foreman
"""
import json
import numpy as np


# %%  json custom encoder to handle complex numpy arrays
def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return np.array(dct['__ndarray__'], dct['dtype']).reshape(dct['shape'])

    if isinstance(dct, dict) and '__ndarraycr__' in dct:
        return np.array(dct['__ndarraycr__'] + 1j * dct['__ndarrayci__'], dct['dtype']).reshape(dct['shape'])

    if isinstance(dct, dict) and '__complexr__' in dct:
        return dct['__complexr__'] + 1j * dct['__complexi__']

    return dct


# Overload dump/load to default use this behavior.
def dumps(*args, **kwargs):
    kwargs.setdefault('cls', NumpyJSONEncoder)
    return json.dumps(*args, **kwargs)


def loads(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.loads(*args, **kwargs)


def dump(*args, **kwargs):
    kwargs.setdefault('cls', NumpyJSONEncoder)
    return json.dump(*args, **kwargs)


def load(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        # print(type(obj))
        if isinstance(obj, np.ndarray):
            data = obj.tolist()

            if 'complex' in str(obj.dtype):  # complex
                return dict(__ndarraycr__=np.real(data),
                            __ndarrayci__=np.imag(data),
                            dtype=str(obj.dtype),
                            shape=obj.shape)
            else:
                return dict(__ndarray__=data,
                            dtype=str(obj.dtype),
                            shape=obj.shape)

        if isinstance(obj, complex):
            return dict(__complexr__=obj.real,
                        __complexi__=obj.imag
                        )

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:21:33 2022

@author: Matthew Foreman
"""
import numpy as np
import h5py


def save_dict_to_hdf5(dic, filename):
    """
    Save a dictionary whose contents are only strings, np.float64, np.int64,
    np.ndarray, and other dictionaries following this structure
    to an HDF5 file. These are the sorts of dictionaries that are meant
    to be produced by the ReportInterface__to_dict__() method.
    """
    h5py.get_config().track_order = True
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def load_dict_from_hdf5(filename):
    """
    Load a dictionary whose contents are only strings, floats, ints,
    numpy arrays, and other dictionaries following this structure
    from an HDF5 file. These dictionaries can then be used to reconstruct
    ReportInterface subclass instances using the
    ReportInterface.__from_dict__() method.
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")

    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if item is None:
            continue

        if isinstance(item, list):
            item = np.array(item)

        if isinstance(item, complex):
            itemd = {'__complex_real__': np.real(item),
                     '__complex_imag__': np.imag(item),
                     }
            item = itemd
        if isinstance(item, np.ndarray) and ('complex' in str(item.dtype)):
            itemd = {'__ndarray_complex_real__': np.real(item),
                     '__ndarray_complex_imag__': np.imag(item),
                     }
            item = itemd
        if isinstance(item, tuple):
            item = np.array(item)  # this is likely to fail for all but simple numeric tuples

        # save strings, numpy.int64, numpy.float64 etc types
        if isinstance(item, (np.int32, np.int64, np.float64, str, float, np.float32, int)):
            h5file[path + key] = item

        # save numpy arrays
        elif isinstance(item, np.ndarray):
            h5file[path + key] = item
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            raise ValueError('Cannot save {} - {} type. '.format(path, type(item)))


#     dt = h5py.string_dtype(encoding='utf-8')
# ds = f.create_dataset('VLDS', (100,100), dtype=dt)

def recursively_load_dict_contents_from_group(h5file, path):
    """
    Load contents of an HDF5 group. If further groups are encountered,
    treat them like dicts and continue to load them recursively.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            if key in ['__complex_real__',
                       '__complex_imag__',
                       '__ndarray__complex_real__',
                       '__ndarray_complex_imag__']:
                continue
            elif isinstance(item[()], bytes):
                ans[key] = str(item[()], 'utf-8')
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            # check for complex datatypes
            if '__complex_real__' in item.keys():
                ans[key] = item['__complex_real__'][()] + 1j * (item['__complex_imag__'][()])
            elif '__ndarray_complex_real__' in item.keys():
                ans[key] = item['__ndarray_complex_real__'][()] + 1j * (item['__ndarray_complex_imag__'][()])
            else:  # other keep going with recursive
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 19:16:07 2022

@author: mforeman
"""

import numpy as np
# import json
from complexnetworklibrary._numpy_json import dump, load, dumps, loads, json_numpy_obj_hook,NumpyJSONEncoder

test = np.random.random((5,5))
testc = test + 1j*np.random.random((5,5))

data = {"data": test,
        "datac": testc}

with open('data/test.json', 'w+') as f:
     dump(data,f)
     
with open('data/test.json') as f:
     data2 = load(f)
     
test2 = data2['data']
testc2 = data2['datac']
 
print(test2==test)
print(testc2 == testc)
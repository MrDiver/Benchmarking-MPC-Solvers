import numpy as np
import pickle
from bson.binary import Binary
import datetime


def encodeDict(dict):
    prepared = {}
    for key, value in dict.items():
        if type(value) == np.ndarray:
            prepared[key] = Binary(pickle.dumps(value, protocol=2))
        else:
            prepared[key] = value
    prepared["date"] = datetime.datetime.now()
    return prepared


def decodeDict(dict):
    prepared = {}
    for key, value in dict.items():
        if type(value) == bytes:
            prepared[key] = pickle.loads(value)
        else:
            prepared[key] = value
    return prepared

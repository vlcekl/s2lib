#!//anaconda/envs/py36/bin/python
#
# File name:   write_params_pickle.py
# Date:        2019/01/26 14:21
# Author:      Lukas Vlcek
#
# Description: 
#

import sys
import re
import pickle

def params_to_pickle(params_dict, params_pickle, option='r+b', protocol=2):#, encoding='latin1'):
    """
    Store hyperparameters and output parameters in a pickle.
    By default, append new parameters to an existing file and create a new one if it does not exist

    Parameters
    ----------
    params_dict: dict
                 dict or parameters and hyperparameters
    params_pickle: str
                 name with full path of the pickle file
    option: str
            write option (i) 'r+b'(default) appends parameters to existing
            file; (ii) 'wb'
    protocol: int
            protocol number for picle storage. Defalut 2 - compatibility with
            Python 2
    """
    
    if not isinstance(params_dict, dict):
        raise ValueError('Parameters not in dict form.')

    params_store = []

    if option == 'r+b':
        try:
            with open(params_pickle, 'rb') as fi:
                params_store.extend(pickle.load(fi))#, encoding=encoding)
        except IOError:
            print('No existing params file, creating a new one.')
    
    params_store.append(params_dict)

    with open(params_pickle, 'wb') as fo:
        pickle.dump(params_store, fo, protocol=protocol)

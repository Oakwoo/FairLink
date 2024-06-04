import json
import numpy as np
import pandas as pd
import networkx as nx
import pickle as pk
from tqdm import tqdm
from texttable import Texttable
import os

from sklearn.linear_model import LogisticRegression



def read_data(args):
    file_ = open(args.folder_path + args.file_name, 'rb')
    data = pk.load(file_)
    file_.close()
    return data

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())


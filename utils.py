import warnings
import os
import sys
import pickle
import pandas as pd

def printProgress(done, total):
    print("Test progress: %d/%d   \r" % (done, total) )

def save_csv(df, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

def save_obj(obj, model_directory, version):
    filename = model_directory + version + '/intent_classifier.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(model_directory, version):
    with open(model_directory + version + '/intent_classifier.pkl', 'rb') as f:
        return pickle.load(f)

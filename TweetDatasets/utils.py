#messy use-case specific utilities
import datetime
from joblib import dump, load
import re, os
import copy
import pandas as pd
import numpy as np
import csv
import gc
import datetime
from copy import deepcopy

#	Custom utilities for reading in dataframes 
#	with column-wise embedding dimensions
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#	Custom utilities to read in and subsample from csv files
def csv_len(fname, dialect='excel'):
  #get length of a csv without opening the entire file
  #	(for large datasets)
  with open(fname, 'r') as f:
    reader = csv.reader(f, dialect=dialect)
    count = 0
    for l in reader:
      count += 1
  return count

def sample_csv(fname, sample_size=None, dialect='excel', row_nums=None, _len=None):
  #getting file len
  if sample_size is None and row_nums is None:
    return pd.read_csv(fname)
  if _len is None:
    _len = csv_len(fname, dialect=dialect)
  if row_nums is not None:
    print('\tcsv has %i rows; taking %i lines' % (_len, len(row_nums)))
    row_nums_ = set(row_nums+[0])
    skip = [ i for i in range(1, _len) if i not in row_nums_ ]
  else:
    print('\tcsv has %i rows; taking %i samples' % (_len, sample_size))
    perm = np.random.permutation(_len-1) #always take the first line
    skip = sorted(perm[sample_size:]+1)
    o = pd.read_csv(fname, skiprows=skip)
  return pd.read_csv(fname, skiprows=skip)

def _procDF(tweet_df, embed_prefix='embed-d', 
            cols_to_include=[ 'Username', 'Datetime', 'Text', 'Links' ]):
  embed_keys = [ key for key in tweet_df.keys() if key.startswith(embed_prefix) ]
  
  tweetsDS_kwargs = {}
  #get embeddings for each tweet from a pandas DataFrame
  #	assumes the format '%s%i' % (embed_prefix, n) 
  #	where n is the embedding dimension
  if len(embed_keys) > 0:
    embed_keys = sorted(embed_keys, key=natural_keys)
    
    embed_list = list(np.array(tweet_df[embed_keys], dtype=np.float32))
    
    tweetsDS_kwargs['Embedding'] = embed_list
    
  for kw in cols_to_include:
    tweetsDS_kwargs[kw] = list(tweet_df[kw])
  
  return tweetsDS_kwargs

#	Core utility to read in csv/dataframe inputs and 
#	transfer them to TweetDataset kwargs
def make_tweetDS_args(fname=None, dataframe=None, sample_size=None, chunk_size=None, chunks=None, 
                      cols_to_include=[ 'Username', 'Datetime', 'Text', 'Links' ]):
  '''Turns a .cvs with tweet data into a dictionary for input 
  into a TweetDataset constructor. Output format is a dictionary
  with keys for Tweet attributes and values as equal-length lists 
  of attr values for each Tweet in the dataset.'''
  assert fname is not None or dataframe is not None, "must pass csv file or a pandas DataFrame"
  assert chunk_size is None or sample_size is None, "cannot both chunk and sample"
  if dataframe is not None:
    tweetsDS_kwargs = _procDF(dataframe)
  elif chunk_size is None:
    df = sample_csv(fname, sample_size=sample_size)
    tweetsDS_kwargs = _procDF(df)
  else:
    _len = csv_len(fname)
    n_chunks = int(np.ceil(_len/chunk_size))
    chunks = [ np.arange(1+i*chunk_size, min(_len, 1+(i+1)*chunk_size)) for i in range(n_chunks) ]
    for i in range(len(chunks)):
      chunk_df = sample_csv(fname, _len=_len, row_nums=chunks[i])
      chunk_dict = _procDF(chunk_df)
      if i == 0: result = chunk_dict
      else: result = _mergeDicts(result, chunk_dict)
    tweetsDS_kwargs = result
  return tweetsDS_kwargs


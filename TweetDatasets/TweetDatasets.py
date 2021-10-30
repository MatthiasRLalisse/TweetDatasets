#class wrappers for tweet datasets
import joblib
import re
import copy
import numpy as np
import linecache
import csv
from .utils import make_tweetDS_args
import collections
try:
  from nltk.tokenize import TweetTokenizer
except ModuleNotFoundError:
  print('to use TweetCollocator functions for ' +
        'gathering unigram and bigram statistics, ' +
        'install nltk.tokenize' )
import pandas as pd
import string

def load(fname):
  return joblib.load(fname)

class Tweet(object):
  #generic class to record Tweet attributes
  #assumes there are attributes Username, Datetime, Text and
  #can be extended with arbitrary attributes by wrapper class
  #TweetDataset
  def __init__(self, **kwargs):
    self.all_attrs = kwargs
    for kw, arg in kwargs.items():
      if kw != 'Text' and type(arg) == str:
        try: arg = eval(arg)
        except: pass
      
      if kw == 'Datetime':	#add a Date field
        setattr(self, 'Date', arg.split()[0])
      setattr(self, kw, arg)
  def __str__(self):
    return ('User:\t\t%s \nDatetime:\t%s\nText:\t\t%s\n' + 
            'Attributes:\t[%s] ' )% (self.Username, self.Datetime, \
                                      self.Text.replace('\n', '[EOL]'), \
                                      ', '.join(list(self.all_attrs.keys())))

class TweetDataset(object):
  '''kwargs are lists (columns) of tweet attributes. protocol is to create a Tweet object for 
  	each row taken across all of the columns. iterate on the zip of all the kw lists
  	and assign {keyword: entry} pairs to the Tweet
  Alternatively, can initialize with the _tweets kwarg, which is a list of 
  	Tweet objects.
  Supports direct indexing, slicing, and indexing with iterables. 
  Attribute setting and getting for the whole dataset refers to Tweet attributes. 
  Example: Running ds.Date returns a list of the Date attribute for each
  	Tweet in dataset ds. Running ds.Attr = Val requires Val to be
  	an iterable with the same length as the dataset, and assigns 
  	Val[i] as an attributes of the Tweet object ds[i].'''
  def __init__(self, _dataframe=None, _csv_fname=None,  _tweets=None, **kwargs):
    #if _tweets kwarg is passed, ignore remaining kwargs
    super(TweetDataset, self).__init__()
    if _tweets is None:
      if _dataframe is not None:
        _kwargs = make_tweetDS_args(dataframe=_dataframe)
      elif _csv_fname is not None:
        _kwargs = make_tweetDS_args(fname=_csv_fname)
      else: _kwargs = {}
      
      kwargs.update(_kwargs)
      keys = list(kwargs.keys())
      val_lists = list(kwargs.values())
      _tweets = []
      for vals in zip(*val_lists):
        tweet_kwargs = { keys[i]: val for i, val in enumerate(vals) }
        _tweets.append(Tweet(**tweet_kwargs))
    else:
      _tweets = list(_tweets)
      assert all([ isinstance(tweet, Tweet) for tweet in _tweets ]), \
             "if initializing with _tweets arg, each item must be a Tweet obj"
    self._tweets = _tweets
    
  def __getattr__(self, attr_name):
    try:
      return vars(self)[attr_name]
    except KeyError:
      return self.__collectattrs__(attr_name)
  
  def __setattr__(self, attr_name, val):
    if '_tweets' in vars(self): #checks if dataset has been initialized
      if attr_name == '_tweets':
        super(TweetDataset, self).__setattr__(attr_name, val)
      else:
        assert len(val) == len(self._tweets), \
               "length of any new dataset attribute must match number of tweets"
        for v, tweet in zip(val, self._tweets):
          setattr(tweet, attr_name, v)
    else:
      #if uninitialized, use default attribute-setting behavior
      super(TweetDataset, self).__setattr__(attr_name, val)
  
  def __getitem__(self, i):
    if isinstance(i, int):
      return self._tweets[i]
    elif isinstance(i, slice):
      sliced_tweets = [ copy.copy(tweet) for tweet in 
                        self._tweets[i.start:i.stop:i.step] ]
      return TweetDataset(_tweets=sliced_tweets)
    else:
      try:
        enum_tweets = [ self._tweets[j] for j in i ]
        return TweetDataset(_tweets=enum_tweets)
      except TypeError:
        pass
    
    #elif isinstance(key, int)
    #return self._tweets[i]
  
  def __len__(self):
    return len(self._tweets)
  
  def __collectattrs__(self, attr_name):
    all_of_attr = []; any_match = False
    is_array = True
    for tweet in self._tweets:
      try: 
        all_of_attr.append(getattr(tweet, attr_name))
        any_match = True
      except AttributeError: 
        all_of_attr.append(None)
    if not any_match:
      if attr_name != '__getstate__':
        print("no tweet has this attribute \"%s\"" % attr_name)
      raise AttributeError
    # if the attribute is array-able, return as an array
    try:
      return np.array(all_of_attr)
    except:
      return all_of_attr
  
  def __getstate__(self):
    return vars(self)
  
  def __setstate__(self, state):
    vars(self).update(state)
  
  def __iadd__(self, b):
    assert isinstance(b, TweetDataset)
    #out_self = copy.deepcopy(self)
    #out_self._tweets += b._tweets
    self._tweets += b._tweets
    return self
  
  def __add__(self, b):
    assert isinstance(b, TweetDataset)
    _cat_tweets = self._tweets + b._tweets
    return TweetDataset(_tweets=_cat_tweets)
  
  def save(self, fname):
    if not fname.endswith('.tds'):
      fname += '.tds' #.tds as canonical "TweetDataset" file extension
    joblib.dump(self, fname)
  
  def apply_filter(self, boolean_filter):
    #boolean_filter is a boolean array or a boolean function
    #e.g. boolean_filter = lambda tweet: True returns the original ds
    if type(boolean_filter) == type(lambda f: f):
      out_tweets = [ copy.copy(tweet) for tweet in self._tweets \
                                     if boolean_filter(tweet) ]
    else:
      out_tweets = [ copy.copy(tweet) for (b,tweet) in 
                           zip(boolean_filter, self._tweets) if b ]
    return TweetDataset(_tweets=out_tweets)
  
  def sample(self, sample_size):
    perm = np.random.permutation(len(self))
    return self[perm[:sample_size]]
  
  def sortby(self, attr_name):
    attr_vals = getattr(self, attr_name)
    sorted_ids = sorted(range(len(self)), key=lambda i: attr_vals[i])
    return self[sorted_ids]
  
  def to_df(self, omit_attrs=['Links']):
    #turn tweet dataset into a dataframe omitting omit_attrs fields
    #omits the Links (hyperlinks in tweet) by default
    if type(omit_attrs) == str: 
      omit_attrs = [omit_attrs]
    #gather all field names
    import pandas as pd
    field_names = set()
    for tweet in self:
      tweet_attrs = vars(tweet).keys()
      incl_fields = [ k for k in tweet_attrs if k not in omit_attrs ]
      field_names = field_names.union(set(incl_fields))
    df_dict = { field: [] for field in field_names }
    for tweet in self:
      for field in df_dict:
        try: df_dict[field].append(getattr(tweet, field))
        except AttributeError: df_dict[field].append(None)
    return pd.DataFrame(df_dict)

class TweetCollocator(object):
  def __init__(self):
    self.tokenizer = TweetTokenizer()
  def fit(self, tweet_ds, in_place=True, vocab=None):
    #fit the "language model" parameters
    #if vocab is not None, do add-one smoothing for everything in the vocab
    #	vocab should be a dictionary: { 'unigram_vocab': list, 'bigram_vocab': list }
    bigrams = collections.defaultdict(int)
    unigrams = collections.defaultdict(int)
    tweet_tokens = []
    punct = re.compile('[%s’“]+' % string.punctuation)
    num = re.compile('([%s]|[0-9]){3,}' % string.punctuation)
    self.n_unigrams, self.n_bigrams = 0, 0
    
    if vocab is not None:
      #add-one smoothing
      for unigram in vocab['unigram_vocab']: unigrams[unigram] += 1
      for bigram in vocab['bigram_vocab']: bigrams[bigram] += 1
      self.n_unigrams += len(vocab['unigram_vocab'])
      self.n_bigrams += len(vocab['bigram_vocab'])
    
    for tweet in tweet_ds.Text:
      tokens = self.proc_tokens(self.tokenizer.tokenize(tweet))
      self.n_unigrams += len(tokens)
      self.n_bigrams += len(tokens) - 1
      for i, token in enumerate(tokens):
        unigrams[token] += 1
        if i <= len(tokens) - 2:
          #only add bigrams where neither gram is punctuation or numerals
          gram1, gram2 = tokens[i:i+2]
          if not(punct.fullmatch(gram1) or punct.fullmatch(gram2) or \
                 num.fullmatch(gram1) or num.fullmatch(gram2)):
            bigrams['%s %s' % (gram1, gram2)] += 1
      tweet_tokens.append(tokens)
    self.unigram_frequencies = dict(unigrams)
    self.bigram_frequencies = dict(bigrams)
    self.ds = (tweet_ds if in_place else tweet_ds[:])
    self.ds.Tokens = tweet_tokens
    
    self.unigram_probabilities = { token: count/self.n_unigrams \
                                          for token, count in \
                                          self.unigram_frequencies.items() }
    self.bigram_probabilities = { token: count/self.n_bigrams \
                                         for token, count in \
                                         self.bigram_frequencies.items() }
    
    self.unigram_logprob = { token: np.log(p) for token, p \
                                    in self.unigram_probabilities.items() }
    self.bigram_logprob = { token: np.log(p) for token, p \
                                   in self.bigram_probabilities.items() }
    
  def proc_tokens(self, tweet_tokens):
    link_regex = re.compile("(?P<url>https?://[^\s]+)")
    def proc_token(token):
      if link_regex.match(token):
        return '[LINK]'
      elif False:
        #place other conditions here
        return ''
      else:
        return token.lower()
    return [ proc_token(token) for token in tweet_tokens ]
  
  def pmi(self, boolean_filter, as_df=True):
    #boolean_filter should be a tweet-wise 
    #	boolean function or a boolean list/array
    sub_ds = self.ds.apply_filter(boolean_filter)
    
    #initialize a local TweetCollocations object to compute
    #	unigram/biggram probabilities subject to the filter
    #	add in the full vocab to do add-one smoothing 
    #	(gets better pmi values for rare types)
    vocab = { 'unigram_vocab' : self.unigram_frequencies.keys(), \
              'bigram_vocab'  : self.bigram_frequencies.keys() }
    
    _collocator = TweetCollocator()
    _collocator.fit(sub_ds, in_place=False, vocab=vocab)
    
    #if t is the condition that x ∈ sub_ds (the dataset with the filter),
    #	then pmi(x,t) = log{p(x|t)/p(x)} = log{p(x|t)} - log{p(x)}
    unigram_pmi = { token: log_p_x_given_t - self.unigram_logprob[token] \
                                          for token, log_p_x_given_t in 
                                          _collocator.unigram_logprob.items() }
    bigram_pmi = { token: log_p_x_given_t - self.bigram_logprob[token] \
                                          for token, log_p_x_given_t in 
                                          _collocator.bigram_logprob.items() }
    
    if as_df: #return the data as a dataframe with both the
               #	marginal and conditional (log) probabilities
      df_data = collections.defaultdict(list)
      for token in _collocator.unigram_probabilities.keys():
        df_data['token'].append(token); df_data['type'].append('unigram')
        df_data['pmi'].append(unigram_pmi[token])
        df_data['conditional_token_logprob'].append(_collocator.unigram_logprob[token]) 
        #									log{p(x|t)}
        df_data['marginal_token_logprob'].append(self.unigram_logprob[token]) 
        #									log{p(x)}
        df_data['conditional_token_prob'].append(_collocator.unigram_probabilities[token]) 
        #									p(x|t)
        df_data['marginal_token_prob'].append(self.unigram_probabilities[token])
        #									p(x)
      
      for token in _collocator.bigram_probabilities.keys():
        df_data['token'].append(token); df_data['type'].append('bigram')
        df_data['pmi'].append(bigram_pmi[token])
        df_data['conditional_token_logprob'].append(_collocator.bigram_logprob[token])
        #									log{p(x|t)}
        df_data['marginal_token_logprob'].append(self.bigram_logprob[token])
        #									log{p(x)}
        df_data['conditional_token_prob'].append(_collocator.bigram_probabilities[token])
        #									p(x|t)
        df_data['marginal_token_prob'].append(self.bigram_probabilities[token])
        #									p(x)
      
      #index with the tokens to do token-hashing (e.g. pmi_df.loc['tent']
      out = pd.DataFrame(data=df_data, index=df_data['token'])
      return out.sort_values(['pmi'], ascending=False)
    else:
      return { 'unigram_pmi': unigram_pmi, 'bigram_pmi': bigram_pmi }



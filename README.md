``TweetDatasets`` provides a set of general-purpose utility classes and functions for storing and processing Twitter data. It is written so as to be object-oriented programming-friendly, and to work particularly well with embeddings and other dataset attributes useful in machine learning.

Usage
===========
The canonical import statement is:
~~~
>>> import TweetDatasets as tweets
~~~
To read in a dataset from a ``.csv`` file, you can input the file as a keyword in initializing a ``TweetDataset``, or by reading it in as a ``pandas`` dataframe and passing that as an argument to the initialization call. A sample of 500 tweets from @FoxNews is provided.
~~~
>>> import pandas as pd
>>> 
>>> csv_filename = 'fox_tweets_sample.csv'
>>> df = pd.read_csv(csv_filename)
>>> fox_tweets = tweets.TweetDataset(_csv_fname=csv_filename)
>>> 
>>> ### or alternatively:
>>> #df = pd.read_csv(csv_filename)
>>> #fox_tweets = tweets.TweetDataset(_dataframe=df)
~~~
The tweet attributes are then accessible using object-oriented syntax, where dataframe column keys become the attribute names. TweetDatasets are directly indexable:
~~~
>>> print(fox_tweets[8].Username, fox_tweets[8].Datetime, '\n', fox_tweets[8].Text)
FoxNews 2021-09-16 02:20:00+00:00 
 Bongino: If Milley warned China, 'he should be court-martialed'
https://t.co/iCM38pe5JI
~~~
The ``print`` function assumes the tweet has ``Username, Datetime,`` and ``Text`` attributes.
~~~
>>> print(fox_tweets[8])
User:		FoxNews 
Datetime:	2021-09-16 02:20:00+00:00
Text:		Bongino: If Milley warned China, 'he should be court-martialed'[EOL]https://t.co/iCM38pe5JI
Attributes:	[Embedding, Username, Datetime, Text, Links] 
~~~
As an alternative to a ``.csv`` or ``pandas`` input, you can initialize with custom keyword arguments. The input format is a set of keyword args whose values are lists (or other iterable) of attribute values, one for each tweet. The keyword becomes the attribute name.
~~~
>>> from datetime import datetime
>>> import numpy as np
>>> 
>>> my_tweets = tweets.TweetDataset(
...                    Username=['MatthiasLalisse']*3, 
...                    Text=[   'Hello world', 
...                             '@jack Thanks for all the tweets', 
...                             '@Sen_JoeManchin What\'s the hold-up?' ], 
...                    Datetime=[str(datetime.now())]*3, 
...                    Likes=[4, 889, 45], 
...                    RandomAttr=np.random.randn(3)
...                    )
>>> 
>>> print(my_tweets[1]) #assigned attributes also appear in dir(my_tweets)
User:		MatthiasLalisse 
Datetime:	2021-10-29 19:57:08.756721
Text:		@jack Thanks for all the tweets
Attributes:	[Username, Text, Datetime, Likes, RandomAttr] 
~~~
You can get a given attribute for all tweets by referencing the entire dataset. Similarly, you can assign arbitrary attributes on a tweet-by-tweet basis, or by assigning it to the entire dataset as a list/array of values, which must match the length of the dataset.
~~~
>>> print(my_tweets.Text)
['Hello world' '@jack Thanks for all the tweets'
 "@Sen_JoeManchin What's the hold-up?"]
>>> 
>>> my_tweets.SentimentLabel = [ 'Neu', 'Pos', 'Neg' ]
>>> print(my_tweets[2].SentimentLabel)
Neg
~~~
The ``TweetDataset`` class supports slicing and indexing with integer arrays.
~~~
>>> len(fox_tweets[25:115])
90
>>> random_sample = np.random.choice(500, size=90, replace=False)
>>> len(fox_tweets[random_sample])
90
~~~
To filter the dataset with a boolean function, use the ``apply_filter`` method.
~~~
>>> #get all tweets about Trump
>>> trump_tweets = fox_tweets.apply_filter(lambda tweet: 'Trump' in tweet.Text)
>>> print(trump_tweets[3])
User:		FoxNews 
Datetime:	2021-09-15 04:20:00+00:00
Text:		The Breakfast Club hits Joy Reid for vaccine-scolding Nicki Minaj, invokes MSNBC host's skepticism under Trump[EOL]https://t.co/92FCGC8qO5
Attributes:	[Embedding, Username, Datetime, Text, Links]
~~~

Working with Embeddings
=============================
These data structures were primarily built with vectorization in mind. The sample in ``fox_tweets_sample.csv`` includes tweet vectors embedded using [BERTweet](https://github.com/VinAIResearch/BERTweet).
~~~
>>> print(fox_tweets.Embedding)
[[ 0.224  -0.1174  0.0786 ... -0.0399 -0.1419 -0.1405]
 [ 0.3698 -0.1038 -0.0017 ... -0.0494 -0.1326 -0.086 ]
 [ 0.3263 -0.0727 -0.0244 ... -0.0782 -0.1216 -0.1455]
 ...
 [ 0.2859 -0.2177 -0.0268 ... -0.0715 -0.1044 -0.1045]
 [ 0.3055 -0.1327 -0.0481 ... -0.0511 -0.112  -0.0925]
 [ 0.2976 -0.1252  0.0245 ... -0.0761 -0.1167 -0.1192]]
>>> print(fox_tweets.Embedding.shape)
(500, 768)
~~~
You can also design your own model. Say you have a module called ``glove`` that provides a dictionary of ``GloVe`` vectors for English words. Define a Bag-of-Words function to combine the vectors via summation, and then assign the result to the dataset.
~~~
>>> import glove, re, string
>>> word2vec = glove.GloveVec(dim=50)
>>> 
>>> #remove punctuation
>>> punk = '[%s]' % string.punctuation
>>> 
>>> def BOW(tweet):
...   tweet_words = [ string_ for string_ in re.sub(punk, ' ', 
...                        tweet.Text.lower()).split() if string_ in word2vec ]
...   return np.sum([ word2vec[word] for word in tweet_words ], axis=0)
... 
>>> trump_tweets.BagEmbedding = [ BOW(tweet) for tweet in trump_tweets ]
~~~
Appropriately typed attributes of the right shape are automatically cast to a ``numpy`` array.
~~~
>>> print(trump_tweets.BagEmbedding)
[[-0.1002 -1.0167  1.6337 ...  3.7307 -0.3411  9.398 ]
 [ 1.4908  4.303   4.7031 ... -4.4536  3.5747  1.9136]
 [ 4.4885  4.7253  3.1226 ...  1.3031  0.4088  4.3161]
 ...
 [ 4.4754  3.7761  4.8814 ...  1.8662 -1.0883  3.2817]
 [ 4.175   1.1041  5.8511 ...  1.0399  1.7365  4.6983]
 [ 5.8879  2.5091  8.3648 ... -1.9071 -2.3973  0.9574]]
>>> print(trump_tweets.BagEmbedding.shape)
(18, 50)
~~~
To save a model, use the ``save`` method and ``load`` function, which are just wrappers for pickle.
~~~
>>> trump_tweets.save('trump_tweets.tds')
>>> my_dataset = tweets.load('trump_tweets.tds')
~~~



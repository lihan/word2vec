# word2vec

Word2Vec implementation in Tensorflow.

### Uses:

By default, it teads all the text file under `./text` directory

```python

from word2vec import Word2Vec

word2vec = Word2Vec(128, 4)
word2vec.fit()

```

### Todos:

* Implement `transform` method to output similar words 
* Phrase detection before training
* Subsample less frequent words
* Add code examples
import os
import re
import math
import pickle
import string
import numpy as np
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.util import skipgrams
from tensorflow.contrib.tensorboard.plugins import projector
from functools import reduce
from multiprocessing import Pool


pun_regex = re.compile('[%s]' % re.escape(string.punctuation))


def normalise_line(line):
    _sentences = []
    sent_tokenise_list = sent_tokenize(line)
    for _sent in sent_tokenise_list:
        _sen = pun_regex.sub('', _sent.lower())
        _sen = re.sub(r'\s+', ' ', _sen)
        _sentences.append(_sen)
    return _sentences


def get_word_count(text_file_path):
    counter = Counter()
    with open(text_file_path, 'r', errors='ignore') as _file:
        for paragraph in _file:
            lines = normalise_line(paragraph)
            for line in lines:
                counter.update(line.split())
    return counter


class Word2Vec:
    def __init__(self,
                 embedding_size,
                 context_size,
                 text_file_dir='text',
                 max_vocab_size=100000,
                 min_occurrence=2,
                 learning_rate=10e-5,
                 batch_size=512,
                 valid_size=10,
                 epochs=20):
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.max_vocab_size = max_vocab_size
        self.min_occurrence = min_occurrence
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.epochs = epochs
        self.text_file_dir = text_file_dir

        self._word_to_index = {}
        self._index_to_word = {}

    def fit(self):
        self._fit_to_corpus()
        self._build_graph()
        self._train()

    def _get_text_file_paths(self):
        text_file_path = os.path.join(os.path.dirname(os.path.realpath(__name__)), self.text_file_dir)
        return map(lambda x: os.path.join(text_file_path, x), os.listdir(text_file_path))

    def _indexing_corpus(self):
        word_to_index = {}
        index_to_word = {}
        text_files = self._get_text_file_paths()

        pool = Pool()
        counters = pool.map(get_word_count, text_files)
        counter = reduce(lambda x, y: x + y, counters)
        vocab = counter.most_common(self.max_vocab_size)

        for i, (word, _c) in enumerate(vocab):
            word_to_index[word] = i
            index_to_word[i] = word

        last_index = len(word_to_index)
        word_to_index['UNKNOWN'] = last_index
        index_to_word[last_index] = 'UNKNOWN'
        self.vocab_size = len(word_to_index)

        with open('word_to_index.pickle', 'wb') as handle:
            pickle.dump(word_to_index, handle)

        with open('index_to_word.pickle', 'wb') as handle:
            pickle.dump(index_to_word, handle)

        return word_to_index, index_to_word

    def _log_metadata(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        metafilename = os.path.join('logs', "metadata.tsv")
        metafile = open(metafilename, "w")
        metafile.write("label\tid\n")
        for i in range(len(self._index_to_word)):
            metafile.write("%s\t%d\n" % (self._index_to_word[i], i))
        metafile.close()

    def _fit_to_corpus(self):
        self._word_to_index, self._index_to_word = self._indexing_corpus()
        self._log_metadata()

    def _build_graph(self):
        valid_window = 1000

        self._train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_inputs')
        self._train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='train_labels')
        self._valid_examples = np.random.choice(valid_window, self.valid_size, replace=False)
        self._valid_dataset = tf.constant(self._valid_examples, dtype=tf.int32)

        self._embeddings = tf.get_variable('embeddings', initializer=tf.random_uniform(
            [self.vocab_size, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(self._embeddings, self._train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.get_variable('nce_weights',
                                      initializer=tf.truncated_normal(
                                          [self.vocab_size, self.embedding_size],
                                          stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.get_variable('nce_biases', initializer=tf.zeros([self.vocab_size]))

        num_sampled = 64
        with tf.name_scope('Loss'):
            self._loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self._train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=self.vocab_size))

            tf.summary.scalar('loss', self._loss)

        self._optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(self._embeddings), 1, keep_dims=True))
        normalized_embeddings = self._embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, self._valid_dataset)

        self._similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

    def generate_batch(self, file_paths, batch_size):
        samples = []
        labels = []
        unknown = self._word_to_index['UNKNOWN']
        for file_path in file_paths:
            with open(file_path, 'r', errors='ignore') as _file:
                for paragraph in _file:
                    sentences = normalise_line(paragraph)
                    for sentence in sentences:
                        for sample, label in skipgrams(sentence.split(), 2, self.context_size):
                            samples.append(self._word_to_index.get(sample, unknown))
                            labels.append([self._word_to_index.get(label, unknown)])

                            if len(samples) == batch_size:
                                yield (samples, labels,)
                                samples = []
                                labels = []

    def _train(self):
        text_file_paths = self._get_text_file_paths()
        summary_merged = tf.summary.merge_all()
        # Create Projector config
        config = projector.ProjectorConfig()
        # Add embedding visualizer
        embedding = config.embeddings.add()
        # Attache the name 'embedding'
        embedding.tensor_name = self._embeddings.name
        # Metafile which is described later
        embedding.metadata_path = "metadata.tsv"

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # We must initialize all variables before we use them.
            init.run()
            train_writer = tf.summary.FileWriter('logs', session.graph)
            # Add writer and config to Projector
            projector.visualize_embeddings(train_writer, config)

            average_loss = 0

            for epoch in range(self.epochs):
                num_steps = self.vocab_size // self.batch_size
                for step, (batch_inputs, batch_labels) in enumerate(
                    self.generate_batch(text_file_paths, self.batch_size)):
                    global_step = epoch * num_steps + step
                    feed_dict = {self._train_inputs: batch_inputs, self._train_labels: batch_labels}

                    # We perform one update step by evaluating the optimizer op (including it
                    # in the list of returned values for session.run()
                    _, loss_val, _summary_loss = session.run([
                        self._optimizer, self._loss, summary_merged], feed_dict=feed_dict)
                    train_writer.add_summary(_summary_loss, global_step=global_step)
                    average_loss += loss_val

                    if step % 2000 == 0:
                        if step > 0:
                            average_loss /= 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step ', step, ': ', average_loss)
                        average_loss = 0

                    # Note that this is expensive (~20% slowdown if computed every 500 steps)
                    if step % 10000 == 0:
                        sim = self._similarity.eval()
                        for i in range(self.valid_size):
                            valid_word = self._index_to_word[self._valid_examples[i]]
                            top_k = 8  # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = self._index_to_word[nearest[k]]
                                log_str = '%s %s,' % (log_str, close_word)
                            print(log_str)
                        # Save the model
                        saver = tf.train.Saver()
                        saver.save(session, os.path.join('logs', 'model.ckpt'), global_step)

            train_writer.close()

    @property
    def embeddings(self):
        return self._embeddings

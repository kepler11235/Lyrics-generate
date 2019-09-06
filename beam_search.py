import numpy as np
import zipfile
import tensorflow as tf
import os, collections, re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
num_unroll = 120
emb_size = 512
layers = [600, 700, 800]
hidden_size = layers[-1]
beam_width, length = 10, 6


def beam_next(ind, state, size, seq):
    last_word = ind // size
    ind %= size
    n_state = tuple([s[last_word] for s in state])
    if len(seq) is 0:
        t_seq = np.expand_dims(ind, axis=1)
    else:
        t_seq = []
        for step, i in enumerate(last_word):
            t_seq.append(list(seq[i]))
            t_seq[step].append(ind[step])
    return np.expand_dims(ind, axis=1), n_state, t_seq


with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ').replace('\u3000', ' ')
corpus_chars = re.sub('[a-zA-Z]〖〗', '', corpus_chars)
all_words = [word for word in corpus_chars]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
idx_to_char, _ = zip(*count_pairs)

char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)

head = '你说'
songs = list(head)
headvecs = np.array([char_to_idx[h] for h in head])
outW = np.expand_dims(np.array([char_to_idx[head[-1]]]), axis=0)
headvecs = np.expand_dims(headvecs, axis=0)
outW = np.tile(outW, [beam_width, 1])

X = tf.placeholder(tf.int32, [None, None])
embedding = tf.Variable(tf.truncated_normal([vocab_size, emb_size], stddev=0.1, dtype=tf.float32))
X_ = tf.nn.embedding_lookup(embedding, X)
if_beam = tf.placeholder(tf.bool)

cell = [tf.nn.rnn_cell.GRUCell(num_units=layer) for layer in layers]
cells = tf.nn.rnn_cell.MultiRNNCell(cell)
state = cells.zero_state(1, dtype=tf.float32)
b_state = cells.zero_state(beam_width, dtype=tf.float32)

output, a_hidden = tf.cond(if_beam, lambda: tf.nn.dynamic_rnn(cells, X_, initial_state=b_state),
                           lambda: tf.nn.dynamic_rnn(cells, X_, initial_state=state))
hidden = a_hidden[-1]

W = tf.Variable(tf.random_normal([layers[-1], vocab_size], stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.random_normal([vocab_size], stddev=0.1, dtype=tf.float32))
logit = tf.matmul(hidden, W) + b

init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init.run()
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, 'my_model/final_model.ckpt')
    initial_state = sess.run(state)
    start_H = sess.run(a_hidden, feed_dict={X: headvecs, state: initial_state, if_beam: False})
    start_H = tuple([np.tile(s, [beam_width, 1]) for s in start_H])
    log_sum_prob = tf.placeholder(tf.float32, [None, ])
    log_sum_prob_ = tf.squeeze(tf.expand_dims(log_sum_prob, axis=1) + logit)
    log_sum_prob_ = tf.squeeze(tf.reshape(log_sum_prob_, [1, -1]))
    log_sum_prob_, index = tf.nn.top_k(log_sum_prob_, beam_width)
    saved_prob = np.log([1.0] + [0.0] * (beam_width - 1))
    word_list = []
    for i in range(num_unroll):
        if i % length == 0:
            if i != 0:
                word_list.append(current[0])
            current = []
        [saved_prob, ind, start_H] = sess.run([log_sum_prob_, index, a_hidden],
                                              feed_dict={X: outW, b_state: start_H, log_sum_prob: saved_prob,
                                                         if_beam: True})
        outW, start_H, current = beam_next(ind, start_H, vocab_size, current)
    for i in word_list:
        for word in i:
            head += str(idx_to_char[word])
    print(head)

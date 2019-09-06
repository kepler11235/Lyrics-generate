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
# corpus_indices = [char_to_idx[char] for char in corpus_chars]

head = '你说是'
songs = list(head)
headvecs = np.array([char_to_idx[h] for h in head])
outW = np.array([char_to_idx[head[-1]]])

X = tf.placeholder(tf.int32, [None])
# X_ = tf.one_hot(X, depth=vocab_size, dtype=tf.float32)

embedding = tf.Variable(tf.truncated_normal([vocab_size, emb_size], stddev=0.1, dtype=tf.float32))
X_ = tf.nn.embedding_lookup(embedding, X)
X_ = tf.expand_dims(X_, dim=0)

cell = [tf.nn.rnn_cell.GRUCell(num_units=layer) for layer in layers]
cells = tf.nn.rnn_cell.MultiRNNCell(cell)
state = cells.zero_state(1, dtype=tf.float32)

output, a_hidden = tf.nn.dynamic_rnn(cells, X_, initial_state=state)
hidden = a_hidden[-1]

W = tf.Variable(tf.random_normal([layers[-1], vocab_size], stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.random_normal([vocab_size], stddev=0.1, dtype=tf.float32))
logit = tf.matmul(hidden, W) + b
dropout_rate = tf.placeholder(tf.float32)
logit = tf.nn.dropout(logit, dropout_rate)
out = tf.argmax(logit, 1)

init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init.run()
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, 'my_model/final_model.ckpt')
    initial_state = sess.run(state)
    start_H = sess.run(state, feed_dict={X: headvecs, state: initial_state, dropout_rate: 0.98})
    for i in range(num_unroll):
        [outW, start_H] = sess.run([out, a_hidden], feed_dict={X: outW, state: start_H, dropout_rate: 0.95})
        songs.append(idx_to_char[outW[0]])
    print(''.join(songs))

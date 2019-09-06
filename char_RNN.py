import numpy as np
import random
import zipfile
import tensorflow as tf
import os, re
import collections

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
num_unroll = 32
batch_size = 32
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

# idx_to_char = list(set(corpus_chars))
# char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
corpus_indices = [char_to_idx[char] for char in corpus_chars]


def data_iter_random(corpus_indices, batch_size, num_steps):
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X), np.array(Y)


X = tf.placeholder(tf.int32, [None, num_unroll])
# X_ = tf.one_hot(X, depth=vocab_size)
Y = tf.placeholder(tf.int32, [None, num_unroll])

embedding = tf.Variable(tf.truncated_normal([vocab_size, emb_size], stddev=0.1, dtype=tf.float32))
X_ = tf.nn.embedding_lookup(embedding, X)

cell = [tf.nn.rnn_cell.GRUCell(num_units=layer) for layer in layers]
cells = tf.nn.rnn_cell.MultiRNNCell(cell)
output, hidden = tf.nn.dynamic_rnn(cells, X_, dtype=tf.float32)
output = tf.reshape(output, [-1, hidden_size])

W = tf.Variable(tf.random_normal([hidden_size, vocab_size], stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.random_normal([vocab_size], stddev=0.1, dtype=tf.float32))
logit = tf.matmul(output, W) + b
dropout = tf.placeholder(tf.float32)
logit = tf.nn.dropout(logit, dropout)

logit = tf.reshape(logit, [-1, num_unroll, vocab_size])
Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logit)
loss = tf.reduce_mean(Xentropy)
pp = tf.reduce_mean(tf.exp(Xentropy))
optimizer = tf.train.AdamOptimizer()
grads = optimizer.compute_gradients(loss)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 2), v)
training_op = optimizer.apply_gradients(grads)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    init.run()
    for epoch in range(135):
        for batch_x, batch_y in data_iter_random(corpus_indices, batch_size, num_unroll):
            training_op.run(feed_dict={X: batch_x, Y: batch_y, dropout: 0.7})
            # print(logit.eval(feed_dict={X: batch_x}).shape)
        ppp = pp.eval(feed_dict={X: batch_x, Y: batch_y, dropout: 1})
        print(epoch, ':', ppp)
        if ppp < 1.9:
            break
    save_path = saver.save(sess, 'my_model/final_model.ckpt')

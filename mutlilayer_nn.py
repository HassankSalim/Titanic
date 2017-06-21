import tensorflow as tf
from preprocess import processCatData, processNumData, divideDataset
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# if len(argv) < 2:
#     print('Usage : python train_nn.py freeze_file.pb')
#     exit()
# else:
#     pickle_file_name = argv[1]
# ohe.transform(np.array(j).reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = divideDataset()
train_set = np.hstack((processCatData(X_train), processNumData(X_train)))
validation_set = np.hstack((processCatData(X_test), processNumData(X_test)))

feature_size = train_set.shape[1]
n_class = 2
display_step = 1
training_iters = 10

ohe = OneHotEncoder()
ohe.fit([[0], [1]])

Batch_Size  = 50
def mNext():
    zipped_train = zip(train_set, y_train)
    np.random.shuffle(zipped_train)
    j = 0
    size = X_train.shape[0]
    while j+Batch_Size < size:
         data, ground_truth = zip(*zipped_train[j : j+Batch_Size])
         j += Batch_Size
         encoded_ground__truth = ohe.transform(np.array(ground_truth).reshape(-1, 1)).toarray()
         reformat_data = np.vstack(data)
         yield reformat_data, encoded_ground__truth
    data, ground_truth = zip(*zipped_train[j:])
    encoded_ground__truth = ohe.transform(np.array(ground_truth).reshape(-1, 1)).toarray()
    reformat_data = np.vstack(data)
    yield reformat_data, encoded_ground__truth

print(feature_size)
# Creating network

X = tf.placeholder(tf.float32, shape=[None, feature_size])
y = tf.placeholder(tf.float32, shape=[None, n_class])

weights = {
    'input_hidden_w1':tf.Variable(tf.truncated_normal(shape=[feature_size, 5])),
    'hidden_out_w2':tf.Variable(tf.truncated_normal(shape=[5, n_class]))
}

bias = {
    'input_hidden_b1':tf.Variable(tf.truncated_normal(shape=[5])),
    'hidden_out_b2':tf.Variable(tf.truncated_normal(shape=[n_class]))
}

def feedfwd():
    a1 = tf.add(tf.matmul(X, weights['input_hidden_w1']), bias['input_hidden_b1'])
    z1 = tf.nn.relu(a1)
    a2 = tf.add(tf.matmul(z1, weights['hidden_out_w2']), bias['hidden_out_b2'])
    z2 = tf.nn.relu(a2)
    return z2

pred = feedfwd()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer().minimize(cost)
corret_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(corret_pred, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as  sess:
    sess.run(init_op)
    for count in range(training_iters):
        for x_batch, y_batch in mNext():
            _, loss = sess.run([train_op, cost], feed_dict={X:x_batch, y:y_batch})
            print('Training loss %.4f'%loss)
        if count % display_step == 0:
            accuracy_score = sess.run(acc, feed_dict={X:x_batch, y:y_batch})
            print('Accuracy %.4f after iteration %d'%(accuracy_score, count))
        print('Completed iteration')

exit()

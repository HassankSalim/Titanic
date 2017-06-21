import tensorflow as tf
from preprocess import processCatData, processNumData, divideDataset
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder
<<<<<<< HEAD
=======
from sys import argv
from tensorflow.python.framework import graph_util
>>>>>>> 26737a4... Modify multilayer_nn.py to save model and store/freeze model in ProtoBuffer(pb) format

warnings.filterwarnings('ignore')

# if len(argv) < 2:
#     print('Usage : python train_nn.py freeze_file.pb')
#     exit()
# else:
#     pickle_file_name = argv[1]
# ohe.transform(np.array(j).reshape(-1, 1)).toarray()

display_step = 1
<<<<<<< HEAD
training_iters = 10
=======
training_iters = 1000
>>>>>>> 26737a4... Modify multilayer_nn.py to save model and store/freeze model in ProtoBuffer(pb) format

ohe = OneHotEncoder()
ohe.fit([[0], [1]])


X_train, X_test, y_train, y_test = divideDataset()
train_set = np.hstack((processCatData(X_train), processNumData(X_train)))
validation_set = np.hstack((processCatData(X_test), processNumData(X_test)))

y_test = ohe.transform(np.array(y_test).reshape(-1, 1)).toarray()
validation_set = np.vstack(validation_set)

feature_size = train_set.shape[1]
n_class = 2

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

# Function to freeze model (Thnks to meta-ai)
def freeze_graph(model_folder, acc):

    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + model_name+ "_" + acc + ".pb"

    output_node_names = "final_output"

    clear_devices = True

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

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
output = tf.argmax(pred, 1, name='final_output')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer().minimize(cost)
corret_pred = tf.equal(output, tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(corret_pred, tf.float32))
acc_for_filename = 0
init_op = tf.global_variables_initializer()

with tf.Session() as  sess:
    sess.run(init_op)
    for count in range(training_iters):
        for x_batch, y_batch in mNext():
            sess.run([train_op, cost], feed_dict={X:x_batch, y:y_batch})
        if count % display_step == 0:
            accuracy_score, loss = sess.run([acc, cost], feed_dict={ X : validation_set, y : y_test })
            print('Accuracy %.4f and Training loss %.4f after iteration %d'%(accuracy_score, loss, count))
<<<<<<< HEAD
    print('Finished training model')
    saver = tf.train.Saver()
=======
            acc_for_filename = str(int(accuracy_score * 100))
    saver = tf.train.Saver()
    saver.save(sess, 'models/' + model_name + '.ckpt')
    print('Finished training model and saved')
freeze_graph('models', acc_for_filename)
>>>>>>> 26737a4... Modify multilayer_nn.py to save model and store/freeze model in ProtoBuffer(pb) format

exit()

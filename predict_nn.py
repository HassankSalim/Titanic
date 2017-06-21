import tensorflow as tf
from sys import argv
import numpy as np
import pandas as pd
from preprocess import processNumData, processCatData, divideDataset, get_X_y

if len(argv) < 3:
    print('Usage : python train_nn.py model_name output_csv_name')
    exit()
else:
    file_name = argv[1]


def load_graph(file_name):
    with tf.gfile.GFile(file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

graph = load_graph(file_name)
X = graph.get_tensor_by_name('prefix/Placeholder:0')
output = graph.get_tensor_by_name('prefix/final_output:0')

test = pd.read_csv('data/test.csv')
output_csv_name = 'submission/' + argv[2]

processed_test_data = np.hstack((processCatData(test), processNumData(test)))
processed_test_data = np.vstack(processed_test_data)

with tf.Session(graph=graph) as sess:
    pred = sess.run(output, feed_dict={ X : processed_test_data })

pred = pd.DataFrame(pred)
pred.columns = ['Survived']

submission = pd.concat([test['PassengerId'], pred], axis=1)
submission.to_csv(output_csv_name, index=False)

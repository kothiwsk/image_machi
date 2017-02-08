import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.platform
import os
import re
from tensorflow.python.platform import gfile
test_dir = '/Users/lno7761/projects/nature_convservancy/test_stg1/'
modelFullPath='output_graph.pb'
labelsFullPath='output_labels.txt'



def extract_features(list_images):
    nb_features = 8
    features = np.empty((len(list_images),nb_features))
    create_graph()
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for ind, image in enumerate(list_images):
            print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
        return features

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():

    list_images = [test_dir+f for f in os.listdir(test_dir) if re.search('jpg|JPG', f)]

    # Creates graph from saved GraphDef.
    create_graph()

    features_test = extract_features(list_images)

    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    print(lines)
    labels = [w.decode("utf-8").rstrip().upper() for w in lines]
    print(labels)
    header = ['image','ALB','BET','DOL','LAG','NOF','OTHER','SHARK','YFT']
    df=pd.DataFrame(features_test,columns=labels)
    df['image']=list_images
    df=df[header]

    df.to_csv("submit.retraininc.csv",index=None)
    #submit = open('submit.retrainedinc.csv','w')
    #submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')

    #for idx, id_n in enumerate(list_images):
    #    probs=['%s' % p for p in list(y_pred[idx, :])]
    #    submit.write('%s,%s\n' % (str(image_id[idx]),','.join(probs)))

if __name__ == '__main__':
    run_inference_on_image()

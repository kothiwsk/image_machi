# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
#from pyimagesearch.score_test import run_inference_on_image
import argparse
import time
import cv2
import os
from os import listdir,walk
from os.path import isfile, join


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.platform
import re
from tensorflow.python.platform import gfile
test_dir = 'D:/Kaggle/comprtition_sealife_conservation/test_stg1/'
modelFullPath='C:/Users/chand/test_tensorflow/classify_image_graph_def.pb'
labelsFullPath='C:/Users/chand/test_tensorflow/imagenet_synset_to_human_label_map.txt'



def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image")
#args = vars(ap.parse_args())


path = 'D:/Kaggle/comprtition_sealife_conservation/train'
#path = "D:\Kaggle\comprtition_sealife_conservation\Code\chandan_directory\image_machi\src\sliding-window"
path1 = 'D:/Kaggle/comprtition_sealife_conservation/'
lst = []
for (dirpath, dirnames, filenames) in walk(path):
	lst.extend(dirnames)
#print lst
for each in lst:
	paths = path + "/" + each
	#print paths
	files = [f for f in listdir(paths) if isfile(join(paths,f))]
#image = numpy.empty(len(files), dtype=object)
	image = []
	for n in files:
		image.append(cv2.imread(join(paths,n)))

create_graph()

with tf.Session() as sess:
	next_to_last_tensor = sess.graph.get_tensor_by_name('softmax:0')
	# load the image and define the window width and height
	#image = cv2.imread(args["image"])
	(winW, winH) = (256, 256)

	# loop over the image pyramid
	for each in image:
		#for resized in pyramid(each, scale=1.5):
		features = []
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(each, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW
			#windows= cv2.imencode(".jpg",window)
			#cv2.imshow("window", window)
			#cv2.waitKey(1)
			number = np.random.uniform(0,1)
			if(number <= 0.1):
				cv2.imwrite(path1 + str(x) + "," + str(y) + ".jpg", window)
			# img_str = cv2.imencode('.jpg', window)[1].tostring()
			# predictions = sess.run(next_to_last_tensor, {"DecodeJpeg/contents:0":img_str})
			# predictions = np.squeeze(predictions)
			# #print(features)
			# #node_lookup = NodeLookup()
            #
			# top_k = predictions.argsort()[-10:][::-1]
			# #
			# # print(top_k)
			# f = open(labelsFullPath, 'rb')
			# lines = f.readlines()
			# labels = [str(w).replace("\n", "") for w in lines]
			# #print(labels)
			# for node_id in top_k:
			# 	human_string = labels[node_id]
			# 	score = predictions[node_id]
			# 	if (score >= 0.10):
			# 		print('%s (score = %.5f)' % (human_string, score))


		#f = open(labelsFullPath, 'rb')
		#lines = f.readlines()
		#print("\n")
		#print(lines)
		#labels = [w.decode("utf-8").rstrip().upper() for w in lines]
		#print(labels)
		# header = ['image','ALB','BET','DOL','LAG','NOF','OTHER','SHARK','YFT']
		# df=pd.DataFrame(features_test,columns=labels)
		# df['image']=list_images
		# df=df[header]

		#df.to_csv("submit.retraininc.csv",index=None)


			# since we do not have a classifier, we'll just draw the window
			#clone = resized.copy()
			#cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			#cv2.imshow("Window", clone)
			#cv2.waitKey(1)
			#time.sleep(0.025)





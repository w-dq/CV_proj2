import os
import utils
import json
import cv2
from sklearn.neighbors import NearestNeighbors

def key_pathches(gray):
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)
	kp,dst = sift.compute(gray,kp)
	return dst


DATA_FOLDER = '../256_ObjectCategories'
subfolders = os.walk(DATA_FOLDER)
dirpath, dirnames, fnames = subfolders.__next__()

with open('categories.json','r',encoding='utf-8') as f:
	labels = json.loads(f.read())

with open('vocab512/vocab64.json','r',encoding='utf-8') as f:
	vocab = json.loads(f.read())
neigh = NearestNeighbors(n_neighbors=len(vocab))
neigh.fit(vocab)

# i=0
for dirpath, dirnames, fnames in subfolders:
	total_ = []
	for f in fnames:
		try:
			gray = cv2.imread(dirpath+'/'+f, cv2.IMREAD_GRAYSCALE)
			patches = key_pathches(gray)
			vector = [0]*len(vocab)
			indices = neigh.kneighbors(patches,n_neighbors=1,return_distance=False)
			for ind in indices:
				vector[ind[0]]+=1
			total_.append([vector,int(f.split('_')[0])])
		except:
			continue

	with open('data-512-64/{}.json'.format(dirpath.strip(' ').split('/')[-1]),'w',encoding='utf-8') as file:
		print('dump:',dirpath)
		json.dump(total_,file)
	# i+=1
	# if i == 5:
	# 	break

import os
import utils
import json
import cv2
from sklearn.neighbors import NearestNeighbors

def key_pathches(gray):
	# gray = cv2.resize(gray,(256,int((len(gray[0])*256)/len(gray))))
	sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05,nfeatures = 60)
	kp = sift.detect(gray, None)
	kp,dst = sift.compute(gray,kp)
	try:
		return dst.tolist()
	except:
		return []
		
	# sift = cv2.xfeatures2d.SIFT_create()
	# kp = sift.detect(gray, None)
	# kp,dst = sift.compute(gray,kp)
	# return dst


DATA_FOLDER = '../256_ObjectCategories'
subfolders = os.walk(DATA_FOLDER)
dirpath, dirnames, fnames = subfolders.__next__()

with open('categories.json','r',encoding='utf-8') as f:
	labels = json.loads(f.read())

with open('vocab64.json','r',encoding='utf-8') as f:
	vocab = json.loads(f.read())
neigh = NearestNeighbors(n_neighbors=len(vocab))
neigh.fit(vocab)

i=0
n=0
for dirpath, dirnames, fnames in subfolders:
	total_ = []
	for f in fnames:
		try:
			vector = [[0]*len(vocab)]*5
			gray = cv2.imread(dirpath+'/'+f, cv2.IMREAD_GRAYSCALE)
			patches = key_pathches(gray)
			try:
				indices = neigh.kneighbors(patches,n_neighbors=1,return_distance=False)
				for ind in indices:
					vector[4][ind[0]]+=1
			except:
				n+=1
				print('no key point!!!!!!!')

			patches = key_pathches(gray[:len(gray)//2,:len(gray[0])//2])
			try:
				indices = neigh.kneighbors(patches,n_neighbors=1,return_distance=False)
				for ind in indices:
					vector[0][ind[0]]+=1
			except:
				print(f+" no spm 0")
				pass

			patches = key_pathches(gray[len(gray)//2:,:len(gray[0])//2])
			try:
				indices = neigh.kneighbors(patches,n_neighbors=1,return_distance=False)
				for ind in indices:
					vector[1][ind[0]]+=1
			except:
				print(f+" no spm 1")
				pass


			patches = key_pathches(gray[:len(gray)//2,len(gray[0])//2:])
			try:
				indices = neigh.kneighbors(patches,n_neighbors=1,return_distance=False)
				for ind in indices:
					vector[2][ind[0]]+=1
			except:
				print(f+" no spm 2")
				pass

			patches = key_pathches(gray[len(gray)//2:,len(gray[0])//2:])
			try:
				indices = neigh.kneighbors(patches,n_neighbors=1,return_distance=False)
				for ind in indices:
					vector[3][ind[0]]+=1
			except:
				print(f+" no spm 3")
				pass

			vec = []
			for j in range(5):
				vec.extend(vector[j])

			total_.append([vec,int(f.split('_')[0])])
		except:
			continue

	with open('data-64/{}.json'.format(dirpath.strip(' ').split('/')[-1]),'w',encoding='utf-8') as file:
		print('dump:',dirpath)
		json.dump(total_,file)
	i+=1
	print(i,' /257')
	# if i == 5:
	# 	break
print(n)





import cv2
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import math
import json
# def get_step_vector(gx,gy):
# 	r = [0,0,0,0,0,0,0,0]
# 	x = 0
# 	y = 0
# 	for i in range(4):
# 		for j in range(4):
# 			x += gx[i,j]
# 			y += gy[i,j]
# 	grad = (x**2+y**2)**0.5
# 	ang = np.math.atan2(y,x)
# 	r[int(ang/(math.pi/4))] = grad
# 	return r

# def get_patch_vector(patch):
# 	kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
# 	kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
# 	gx = cv2.filter2D(patch, -1, kernel_x)
# 	gy = cv2.filter2D(patch, -1, kernel_y)
# 	vec = []
# 	for i in [0,4,8,12]:
# 		for j in [0,4,8,12]:
# 			vec = vec + get_step_vector(gx[i:i+4,j:j+4],gy[i:i+4,j:j+4])
# 	return np.array(vec)

def get_repr(img_name,m=0):
	gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)
	kp,dst = sift.compute(gray,kp)
	return dst.tolist()
	# vec_list = list()
	# print(img_name)
	# source_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
	# x_len, y_len = source_img.shape
	# if not m:
	# 	patches = image.extract_patches_2d(source_img,(16,16),max_patches=float( float((x_len*y_len)/(16*16)) / ( (x_len-15)*(y_len-15) ) ))
	# else:
	# 	patches = image.extract_patches_2d(source_img,(16,16),max_patches=m)

	# for patch in patches:
	# 	vec_list.append(get_patch_vector(patch))
	# return vec_list

def get_model(X,cluster=2,bsize=100):
	kmeans = MiniBatchKMeans(n_clusters=cluster,batch_size=bsize).fit(X)
	print("Converge after {} iterations".format(kmeans.n_iter_))
	return kmeans.cluster_centers_

def preprocess(l):
	categories = dict()
	for i in l:
		key,value = i.strip(' ').split('.')
		categories[key.strip(' ')]=value.strip(' ')
	with open('categories.json','w',encoding='utf-8') as f:
		json.dump(categories,f)
	print('Dump categories!')
	return categories

def vectorize(image_name,vocab,spm=0):
	gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
	vector = None
	if spm:
		pass
	else:
		patches = key_pathches(gray)
		vector = [0]*len(vocab)
		for p in patches:
			vector[get_index(p,vocab)] += 1

	return vector

def key_pathches(gray):
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray, None)
	kp,dst = sift.compute(gray,kp)
	return dst

def get_index(patch,vocab):
	max_dis = 0
	max_idx = 0
	for idx,v in enumerate(vocab):
		dis = euclidean_distance(v,patch)
		if dis>max_dis:
			max_dis = dis
			max_idx = idx
	return max_idx

def euclidean_distance(a,b):
	return math.sqrt(sum( (i[0]-i[1])**2 for i in zip(a,b) ))
	











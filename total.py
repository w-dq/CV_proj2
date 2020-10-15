import os
import utils
import json

DATA_FOLDER = '../256_ObjectCategories'
subfolders = os.walk(DATA_FOLDER)

dirpath, dirnames, fnames = subfolders.__next__()
# categories = utils.preprocess(dirnames)
i=0
total = list()
for dirpath, dirnames, fnames in subfolders:
	for f in fnames:
		try:
			total.extend(utils.get_repr_resize(dirpath+'/'+f))
		except:
			continue
	i+=1
	print("{} out of 257".format(i))
	print(len(total))
	# if i==2:
	# 	break

with open('vocab64.json','w',encoding='utf-8') as file:
	print('dump:64')
	json.dump(utils.get_model(total,64).tolist(),file)

with open('vocab128.json','w',encoding='utf-8') as file:
	print('dump:128')
	json.dump(utils.get_model(total,128).tolist(),file)
import os
import utils
import json

DATA_FOLDER = '../256_ObjectCategories'
subfolders = os.walk(DATA_FOLDER)

dirpath, dirnames, fnames = subfolders.__next__()
# categories = utils.preprocess(dirnames)
i=0
for dirpath, dirnames, fnames in subfolders:
	total = list()
	for f in fnames:
		try:
			total.extend(utils.get_repr_resize(dirpath+'/'+f))
		except:
			continue
	with open('centers256-re/{}.json'.format(dirpath.strip(' ').split('/')[-1]),'w',encoding='utf-8') as file:
		print('dump:',dirpath)
		json.dump(utils.get_model(total,256).tolist(),file)
	i+=1
	print("{} out of 257".format(i))
	# if i==2:
	# 	break


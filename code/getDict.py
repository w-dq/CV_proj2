import os
import utils
import json

DATA_FOLDER = 'centers512'
centers = os.listdir(DATA_FOLDER)
total_ = list()
for file in centers:
	print(file)
	try:
		with open('centers512/' + file,'r',encoding='utf-8') as f:
			c = json.loads(f.read())
		total_.extend(c)
	except:
		continue

dic = utils.get_model(total_,256)
print(dic)
with open('vocab512/vocab256.json','w',encoding='utf-8') as f:
	json.dump(dic.tolist(),f)

dic = utils.get_model(total_,128)
print(dic)
with open('vocab512/vocab128.json','w',encoding='utf-8') as f:
	json.dump(dic.tolist(),f)

dic = utils.get_model(total_,64)
print(dic)
with open('vocab512/vocab64.json','w',encoding='utf-8') as f:
	json.dump(dic.tolist(),f)
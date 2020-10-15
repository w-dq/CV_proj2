import cv2
import os
import utils
import json

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


DATA_FOLDER = 'data256-64-re-spm'
data = os.listdir(DATA_FOLDER)

X = list()
y = list()
X_train = list()
X_test = list()
y_train = list()
y_test = list()
for idx,d in enumerate(data):
	print(d)
	try:
		with open(DATA_FOLDER+'/'+d,'r',encoding='utf-8') as f:
			c = json.loads(f.read())
		X_train.extend(i[0] for i in c[:60])
		X_test.extend(i[0] for i in c[60:])
		y_train.extend(i[1] for i in c[:60])
		y_test.extend(i[1] for i in c[60:])
		# X.extend(i[0] for i in c)
		# y.extend(i[1] for i in c)
	except:
		continue

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
clf_l = make_pipeline(StandardScaler(), SVC(C=1,kernel='linear',decision_function_shape='ovr'))
clf_l.fit(X_train,y_train)
print(clf_l.score(X_test,y_test))

clf_l = make_pipeline(StandardScaler(), SVC(C=1,kernel='rbf',decision_function_shape='ovr'))
clf_l.fit(X_train,y_train)
print(clf_l.score(X_test,y_test))


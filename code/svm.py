import cv2
import os
import utils
import json

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


DATA_FOLDER = 'data-512-64'
data = os.listdir(DATA_FOLDER)

X = list()
y = list()
for d in data[:50]:
	print(d)
	try:
		with open(DATA_FOLDER+'/'+d,'r',encoding='utf-8') as f:
			c = json.loads(f.read())
		X.extend(i[0] for i in c)
		y.extend(i[1] for i in c)
	except:
		continue


parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

clf.fit(X_train,y_train)
print('Best score for data1:', clf.best_score_) 
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
# clf_l = make_pipeline(StandardScaler(), SVC(kernel='linear',decision_function_shape='ovr'))
# clf_l.fit(X_train,y_train)
# print('score:',clf_l.score(X_test,y_test))


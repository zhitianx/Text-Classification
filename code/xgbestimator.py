#encoding:utf-8
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
import numpy
from numpy import float64
from scipy.sparse import coo_matrix
from sklearn import cross_validation,metrics
from xgboost.sklearn import XGBClassifier
from scipy.sparse import csc_matrix
from sklearn.model_selection import GridSearchCV

train_id_to_label = {}
with open("train.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == '1'):
			train_id_to_label[line[0]] = 1.;
		else:			
			train_id_to_label[line[0]] = 0.;
print "finish step 1"

source = open("trainfeature_TFIDF.txt","rb")
words_list = {}
tot_word = 0
numberoftrain = 0
for line in source:
	data = line.split(",")
	length = (len(data) - 2) / 2
	numberoftrain = numberoftrain + 1
	for j in range(0,length):
		if (words_list.has_key(data[j * 2]) == False):
			words_list[data[j * 2]] = tot_word
			tot_word = tot_word + 1
source.close()
print "finish step 2"

numberoftest = 0
source = open("testfeature_TFIDF.txt","rb")
for line in source:
	data = line.split(",")
	length = (len(data) - 1) / 2
	numberoftest = numberoftest + 1
	for j in range(0,length):
		if (words_list.has_key(data[j * 2]) == False):
			words_list[data[j * 2]] = tot_word
			tot_word = tot_word + 1
source.close()
print "finish step 3"
print tot_word

label = []
source = open("trainfeature_TFIDF.txt","rb")
i = 0
element = [1,2]
row = [0,1]
column = [0,1]
X_train = coo_matrix((element,(row,column)),shape = (2,2)).tocsc()
Y_train = []
X_test = coo_matrix((element,(row,column)),shape = (2,2)).tocsc()
Y_test = []
row = []
element = []
column = []
change = 0
print "start"
for line in source:
	data = line.split(",")
	length = (len(data) - 2) / 2
	for j in range(0,length):
		value = (float64)(data[j * 2 + 1])
		c = words_list[data[j * 2]]
		row.append(i)
		column.append(c)
		element.append((value + 1.0) * (value + 0.8))
		#element.append(1.0)
	i = i + 1
	label.append(train_id_to_label[data[length * 2]])
	if (i > numberoftrain * 0.8 and change == 0):
		X_train = coo_matrix((element,(row,column)),shape=(i,tot_word)).tocsc()
		Y_train = label
		label = []
		row = []
		column = []
		element = []
		i = 0
		change = 1
X_test = coo_matrix((element,(row,column)),shape=(i,tot_word)).tocsc()
Y_test = label
y_score = []
source.close()
print "finish step 4"
clf = XGBClassifier(
	objective = 'binary:logistic',
	silent = 1,
	seed = 215,
	learning_rate = 0.05,
	gamma = 0.,
	colsample_bytree = 0.8,
	subsample = 0.8,
	base_score = 0.5,
	max_delta_step = 0,
	min_child_weight = 1,
	max_depth = 5)
for es in range(300,450,50):
	clf = XGBClassifier(
	objective = 'binary:logistic',
	silent = 1,
	seed = 215,
	learning_rate = 0.05,
	gamma = 0.,
	n_estimators = es,
	colsample_bytree = 0.8,
	subsample = 0.8,
	base_score = 0.5,
	max_delta_step = 0,
	min_child_weight = 7,
	max_depth = 42)
	y_score = []
	clf.fit(X_train,Y_train)
	predict_X_test = clf.predict_proba(X_test)
	y_score = []
	for each in predict_X_test:
		y_score.append(each[1])
	print metrics.roc_auc_score(Y_test,y_score)
'''
row = []
column = []
element = []
name = []
source = open("testfeature_TFIDF.txt","rb")
i = 0
for line in source:
	data = line.split(",")
	length = (len(data) - 1) / 2
	for j in range(0,length):
		value = (float64)(data[j * 2 + 1])
		c = words_list[data[j * 2]]
		row.append(i)
		column.append(c)
		element.append((value + 1.0) * (value + 0.8))	
		#element.append(1.0)
	i = i + 1
	name.append(data[length * 2])
feature = coo_matrix((element,(row,column)),shape=(i,tot_word)).tocsc()
source.close()
print "finish step 6"

output = open("result_xgboost.csv","wb")
writer = csv.writer(output)
result = clf.predict_proba(feature)
i = 0
writer.writerow(['id','pred'])
for each in result:
	length = len(name[i])
	writer.writerow([name[i][:length - 1],result[i][1]])
	i = i + 1
output.close()
print "finish step 7"
'''


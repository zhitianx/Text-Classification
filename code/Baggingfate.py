#encoding:utf-8
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
import numpy
from numpy import float64
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import coo_matrix
from sklearn import cross_validation,metrics
from sklearn.ensemble import BaggingClassifier

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
for line in source:
	data = line.split(",")
	length = (len(data) - 2) / 2
	for j in range(0,length):
		if (words_list.has_key(data[j * 2]) == False):
			words_list[data[j * 2]] = tot_word
			tot_word = tot_word + 1
source.close()
print "finish step 2"

source = open("testfeature_TFIDF.txt","rb")
for line in source:
	data = line.split(",")
	length = (len(data) - 1) / 2
	for j in range(0,length):
		if (words_list.has_key(data[j * 2]) == False):
			words_list[data[j * 2]] = tot_word
			tot_word = tot_word + 1
source.close()
print "finish step 3"
print "tot_word = ",tot_word
label = []
row = []
column = []
element = []
source = open("trainfeature_TFIDF.txt","rb")
i = 0
for line in source:
	data = line.split(",")
	length = (len(data) - 2) / 2
	for j in range(0,length):
		value = (float64)(data[j * 2 + 1])
		c = words_list[data[j * 2]]
		row.append(i)
		column.append(c)
		element.append((value + 1.0) * (value + 0.8))
	i = i + 1
	label.append(train_id_to_label[data[length * 2]])
feature = coo_matrix((element,(row,column)),shape=(i,tot_word))
source.close()
print "finish step 4"

X_train,X_test,Y_train,Y_test = train_test_split(feature,label,train_size = 0.8,random_state = 215)
bagging = BaggingClassifier(LogisticRegression(penalty = 'l1',solver = 'liblinear',C = 0.1204,random_state = 215),n_estimators = 4,max_samples = 0.9,max_features = 0.9,random_state = 214)
bagging.fit(X_train,Y_train)
print "finish step 5"
predict_X_test = bagging.predict_proba(X_test)
source = open("bagging_validproba.csv","wb")
writer = csv.writer(source)
for each in predict_X_test:
	writer.writerow([each[1]])
source.close()
y_score = []
for each in predict_X_test:
	y_score.append(each[1])
print metrics.roc_auc_score(Y_test,y_score)

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
	i = i + 1
	name.append(data[length * 2])
feature = coo_matrix((element,(row,column)),shape=(i,tot_word))
source.close()
print "finish step 6"

output = open("result_bagging.csv","wb")
writer = csv.writer(output)
result = bagging.predict_proba(feature)
i = 0
writer.writerow(['id','pred'])
for each in result:
	length = len(name[i])
	writer.writerow([name[i][:length - 1],result[i][1]])
	i = i + 1
output.close()
print "finish step 7"


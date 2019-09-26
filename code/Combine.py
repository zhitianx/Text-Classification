#encoding:utf-8
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
import numpy
from numpy import float64
from sklearn.cross_validation import train_test_split
from scipy.sparse import coo_matrix
from sklearn import cross_validation,metrics

Y_test = []
tot = 0
with open("validlabel.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		Y_test.append((float64)(line[0]))
predict_ada = []
predict_decision = []
predict_bagging = []
predict_rawlr = []
predict_randomforest = []
with open("rawlr_validproba.csv") as f:
	reader = csv.reader(f)
	for line in reader:
		predict_rawlr.append((float64)(line[0]))
		tot = tot + 1
with open("randomforest_validproba.csv") as f:
	reader = csv.reader(f)
	for line in reader:
		predict_randomforest.append((float64)(line[0]))
with open("bagging_validproba.csv") as f:
	reader = csv.reader(f)
	for line in reader:
		predict_bagging.append((float64)(line[0]))

y_score = [(float64)(0.) for i in range(0,tot)]
maxscore = (float64)(0.)
p = [(float64)(0.) for i in range(0,tot)]

step = (float64)(0.01)
number = 100
k = 0

for i in range(0,number + 1):
	for j in range(0,number + 1 - i):
		k = number - i - j
		for x in range(0,tot):
			y_score[x] = step * i * predict_rawlr[x]
			y_score[x] = y_score[x] + step * j * predict_randomforest[x]
			y_score[x] = y_score[x] + step * k * predict_bagging[x]
		cons = metrics.roc_auc_score(Y_test,y_score)
		if (cons > maxscore):
			maxscore = cons
			p[0] = step * i
			p[1] = step * j
			p[2] = step * k
print "maxscore = ",maxscore
name = []
source = open("testfeature_TFIDF.txt","rb")
for line in source:
	data = line.split(",")
	length = (len(data) - 1) / 2
	name.append(data[length * 2])
source.close()

act_ada = []
act_decision = []
act_bagging = []
act_rawlr = []
act_randomforest = []
tot = 0

with open("result_rawlr.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == 'pred'):
			continue
		act_rawlr.append((float64)(line[1]))
		tot = tot + 1
with open("result_randomforest.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == 'pred'):
			continue
		act_randomforest.append((float64)(line[1]))
with open("result_bagging.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == 'pred'):
			continue
		act_bagging.append((float64)(line[1]))
print p[0],p[1],p[2]
output = open("result_combine.csv","wb")
writer = csv.writer(output)
writer.writerow(['id','pred'])
for i in range(0,tot):
	length = len(name[i])
	writer.writerow([name[i][:length - 1],p[0] * act_rawlr[i] + p[1] * act_randomforest[i] + p[2] * act_bagging[i]])
output.close()


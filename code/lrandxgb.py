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


name = []
source = open("testfeature_TFIDF.txt","rb")
for line in source:
	data = line.split(",")
	length = (len(data) - 1) / 2
	name.append(data[length * 2])
source.close()

act_xgb = []
act_rawlr = []
act_randomforest = []
tot = 0

with open("result_bestlr.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == 'pred'):
			continue
		act_rawlr.append((float64)(line[1]))
		tot = tot + 1
with open("result_xgboost.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == 'pred'):
			continue
		act_xgb.append((float64)(line[1]))
with open("result_randomforest.csv","rb") as f:
	reader = csv.reader(f)
	for line in reader:
		if (line[1] == 'pred'):
			continue
		act_randomforest.append((float64)(line[1]))
output = open("result_lrandxgb.csv","wb")
writer = csv.writer(output)
writer.writerow(['id','pred'])
for i in range(0,tot):
	length = len(name[i])
	writer.writerow([name[i][:length - 1],0.04 * act_randomforest[i] + 0.08 * act_rawlr[i] + 0.88 * act_xgb[i]])
output.close()


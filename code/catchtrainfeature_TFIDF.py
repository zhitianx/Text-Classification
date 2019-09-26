#encoding:utf-8
import jieba
import csv
import json
import jieba.analyse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
source = open("train.json","rb")
output = open("trainfeature_TFIDF.txt","wb")
for line in source:
	data = json.loads(line)
	words = jieba.analyse.extract_tags(data["content"],100000,withWeight=True)
	l = []
	for x,y in words:
		l.append(str(x))
		l.append(str(y))
	l.append(str(data["id"]))
	l.append(str(data["label"]))
	output.write(",".join(l) + "\n")

source.close()
output.close()

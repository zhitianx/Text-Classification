#encoding:utf-8
import jieba
import csv
import json
import jieba.analyse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
source = open("test.json","rb")
output = open("testfeature_TFIDF.txt","wb")
for line in source:
	data = json.loads(line)
	words = jieba.analyse.extract_tags(data["content"],100000,withWeight=True)
	l = []
	for x,y in words:
		l.append(str(x))
		l.append(str(y))
	l.append(str(data["id"]))
	output.write(",".join(l) + "\n")

source.close()
output.close()

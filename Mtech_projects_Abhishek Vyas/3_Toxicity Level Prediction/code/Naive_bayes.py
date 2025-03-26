#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:34:37 2019

@author: amrit
"""
import math
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords

def find_max(arr):
    maxi = arr[0]
    for i in range(len(arr)):
        if(arr[i]>maxi):
            maxi = arr[i]
    return(arr.index(maxi))

data = pd.read_csv('/home/amrit/Downloads/jigsaw-toxic-comment-classification-challenge/train.csv')
t_data = pd.read_csv('/home/amrit/Downloads/jigsaw-toxic-comment-classification-challenge/test_labels.csv')
tc_data = pd.read_csv('/home/amrit/Downloads/jigsaw-toxic-comment-classification-challenge/test.csv')
print(tc_data.columns)
print(data.columns)
original = []
computed = []

print(t_data.columns)
ids = list(data['id'])
vocab = []
count_to = 0
count_th = 0
count_ob = 0
count_ins = 0
count_ih =0
count_st = 0
cmnt_text = list(data['comment_text'])
cmnt_text_t = list(tc_data['comment_text'])
toxic = list(data['toxic'])
st = list(data['severe_toxic'])
ob = list(data['obscene'])
th = list(data['threat'])
ins = list(data['insult'])
ih = list(data['identity_hate'])
toxic_t = list(t_data['toxic'])
st_t = list(t_data['severe_toxic'])
ob_t = list(t_data['obscene'])
th_t = list(t_data['threat'])
ins_t = list(t_data['insult'])
ih_t = list(t_data['identity_hate'])
tokeniser = RegexpTokenizer(r'[\w]+[-\']?[\w]*')
toxic_cmnt = defaultdict(list)
s_toxic_cmnt = defaultdict(list)
obscene_cmnt = defaultdict(list)
threat_cmnt = defaultdict(list)
insult_cmnt = defaultdict(list)
idh_cmnt = defaultdict(list)
class_dict = [toxic_cmnt,s_toxic_cmnt,obscene_cmnt,threat_cmnt,insult_cmnt,idh_cmnt]
class_count = [count_to,count_st,count_ob,count_th,count_ins,count_ih]
orig_class = [toxic_t,st_t,ob_t,th_t,ins_t,ih_t]
ckk = [0,0,0,0,0,0]
orig_class_tr = [toxic,st,ob,th,ins,ih]


for i in range(len(ids)):
    if(toxic[i] == 1):
        tokens = tokeniser.tokenize(cmnt_text[i])
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        count_to = count_to+len(tokens)
        for j in tokens:
            if(j not in vocab):
                vocab.append(j)
            value = toxic_cmnt.get(j,"Empty")
            if(value == "Empty"):
                toxic_cmnt[j].append(1)
            else:
                v = value[0]+1
                toxic_cmnt[j].pop()
                toxic_cmnt[j].append(v)
    if(st[i] == 1):
        tokens = tokeniser.tokenize(cmnt_text[i])
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        count_st = count_st+len(tokens)
        for j in tokens:
            if(j not in vocab):
                vocab.append(j)
            value = s_toxic_cmnt.get(j,"Empty")
            if(value == "Empty"):
                s_toxic_cmnt[j].append(1)
            else:
                v = value[0]+1
                s_toxic_cmnt[j].pop()
                s_toxic_cmnt[j].append(v)
    if(ob[i] == 1):
        tokens = tokeniser.tokenize(cmnt_text[i])
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        count_ob = count_ob+len(tokens)
        for j in tokens:
            if(j not in vocab):
                vocab.append(j)
            value = obscene_cmnt.get(j,"Empty")
            if(value == "Empty"):
                obscene_cmnt[j].append(1)
            else:
                v = value[0]+1
                obscene_cmnt[j].pop()
                obscene_cmnt[j].append(v)
    if(th[i] == 1):
        tokens = tokeniser.tokenize(cmnt_text[i])
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        count_th = count_th+len(tokens)
        for j in tokens:
            if(j not in vocab):
                vocab.append(j)
            value = threat_cmnt.get(j,"Empty")
            if(value == "Empty"):
                threat_cmnt[j].append(1)
            else:
                v = value[0]+1
                threat_cmnt[j].pop()
                threat_cmnt[j].append(v)
    if(ins[i] == 1):
        tokens = tokeniser.tokenize(cmnt_text[i])
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        count_ins = count_ins+len(tokens)
        for j in tokens:
            if(j not in vocab):
                vocab.append(j)
            value = insult_cmnt.get(j,"Empty")
            if(value == "Empty"):
                insult_cmnt[j].append(1)
            else:
                v = value[0]+1
                insult_cmnt[j].pop()
                insult_cmnt[j].append(v)
    if(ih[i] == 1):
        tokens = tokeniser.tokenize(cmnt_text[i])
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        count_ih = count_ih+len(tokens)
        for j in tokens:
            if(j not in vocab):
                vocab.append(j)
            value = idh_cmnt.get(j,"Empty")
            if(value == "Empty"):
                idh_cmnt[j].append(1)
            else:
                v = value[0]+1
                idh_cmnt[j].pop()
                idh_cmnt[j].append(v)
for i,j in enumerate(cmnt_text_t):
    ori = []
    class_c = []
    if((toxic_t[i]==0 and st_t[i]==0 and ob_t[i]==0 and th_t[i]==0 and ins_t[i]==0 and ih_t[i]==0) or (toxic_t[i]==-1 and st_t[i]==-1 and ob_t[i]==-1 and th_t[i]==-1 and ins_t[i]==-1 and ih_t[i]==-1) ):
        continue
    else:
        tokens = tokeniser.tokenize(j)
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in words]
        for ck,k in enumerate(class_dict):
            pro = 0
            for t in tokens:
                value = k.get(t,"Empty")
                if(value == "Empty"):
                    v =0
                else:
                    v = value[0]
                pro =pro + math.log10((v+1)/(class_count[ck]+len(vocab)))
            class_c.append(pro) 
            if(orig_class[ck][i] == 1):
                ori.append(ck)
    
        c = find_max(class_c)
        for o in ori:
            computed.append(c)
            original.append(o)

print(accuracy_score(original, computed))
print(confusion_matrix(original, computed))
#print(len(ids),"............",toxic.count(1))
#print("Count of severe toxic : ",st.count(1))
#print("Count of obscene : ",ob.count(1))
#print("Count of threat : ",th.count(1))
#print("Count of insult : ",ins.count(1))
#print("Count of identity hate  : ",ih.count(1))
#print("Total labled data  : ",toxic.count(1)+st.count(1)+ob.count(1)+th.count(1)+ins.count(1)+ih.count(1))
#print(len(toxic_t),"............",toxic_t.count(1))
#print("Count of severe toxic : ",st_t.count(1))
#print("Count of obscene : ",ob_t.count(1))
#print("Count of threat : ",th_t.count(1))
#print("Count of insult : ",ins_t.count(1))
#print("Count of identity hate  : ",ih_t.count(1))
print("Total labled data  : ",toxic_t.count(1)+st_t.count(1)+ob_t.count(1)+th_t.count(1)+ins_t.count(1)+ih_t.count(1))
#print("Total labled data  : ",toxic_t.count(1)+st_t.count(1)+ob_t.count(1)+th_t.count(1)+ins_t.count(1)+ih_t.count(1)+toxic.count(1)+st.count(1)+ob.count(1)+th.count(1)+ins.count(1)+ih.count(1))
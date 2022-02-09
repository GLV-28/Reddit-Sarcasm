import nltk
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from collections import Counter
nltk.download()

##Selencting only the three variables we need
desired_var = ["subreddit","parent_comment"]
RedditData=pd.read_csv("train-balanced-sarcasm.csv",sep=",", dtype="string")
SubDataRed=RedditData[desired_var]

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')
SubDataRed['LabelCate'] = labelencoder.fit_transform(SubDataRed['subreddit'])
SubDataRed['LabelCate'] = SubDataRed['LabelCate'].astype('int8')
#enc_df = pd.DataFrame(enc.fit_transform(SubDataRed[['LabelCate']]).toarray())
#SubDataRed = SubDataRed.join(enc_df)


vectorizer = CountVectorizer()
Conv = vectorizer.fit(SubDataRed["subreddit"])
#Conv.vocabulary_
print(len(Conv.vocabulary_))#14879
Conv = vectorizer.fit(SubDataRed["parent_comment"])
print(len(Conv.vocabulary_))#230422
#vectorizer.get_feature_names_out()


stop_words = set(stopwords.words('english'))
table=str.maketrans('','',string.punctuation)
PreP_data=[]
for a in SubDataRed["parent_comment"]:
	case={}
	input_str=a
	input_str=input_str.lower()
# input_str=input_str.translate(str.maketrans("",""))
# input_str=input_str.translate(string.maketrans("",""), string.punctuation)
	input_str=input_str.strip()
# input_str=[w.translate(table) for w in input_str]
	tokens=word_tokenize(input_str)
	case["token"]=[i for i in tokens if not i in stop_words]
	case["token"]=[i for i in case["token"] if not i in table]
	case["token"]=[i for i in case["token"] if i.isalpha()]
	PreP_data.append(case)
Process_data=pd.DataFrame(PreP_data)
Process_data.to_csv("data_process.csv", sep=';')
Process_data=pd.read_csv("data_process.csv",sep=";")
list_freq=[]
for i in range(len(Process_data)):#range(len(data_procs)):
    dict_freq=Counter(Process_data.iloc[i]["token"].strip('][').replace("'","").split(', '))
    dict_freq['Unnamed: 0'] = i
    list_freq.append(dict_freq)



#
# freq_data=pd.DataFrame(list_freq[0:100000])
# final=pd.DataFrame()
# for i in range(len(list_freq)):
# 	#print(i)
# 	c=pd.DataFrame.from_records(list_freq[i], index=[0])
# 	final.append(c)

list_pd=[]
for i in range(len(list_freq)):
	#print(i)
	c=pd.DataFrame.from_records(list_freq[i], index=[0])
	list_pd.append(c)
merged=pd.DataFrame()
for i in range(len(list_pd)):
	merged=pd.concat([merged,list_pd[i]])
	#if (i + 1) in range(len(list_pd)):
	#	merged=pd.concat(list_pd[i])

merged = pd.concat(list_pd)
merged = pd.DataFrame.from_dict(map(dict,list_pd))

with open("freq_data.json") as jsonFile:
    Freq_json = json.load(jsonFile)
    jsonFile.close()

with open("freq_data.json") as jsonFile:
    Freq_json = json.load(jsonFile)
    jsonFile.close()
for j in range(len(Freq_json)):
    for k,v in list(Freq_json[j].items()):
        if k != "Unnamed: 0":
            if k not in all_wrd:
                del Freq_json[j][k]



for j in range(len(Freq_json)):
    for k,v in list(Freq_json[j].items()):
        if k != "Unnamed: 0":
            if v <= 50:
                del Freq_json[j][k]


filter = [i for i in Freq_json if not (len(i) <= 1)]
filter=pd.DataFrame(filter)
filter=filter.fillna(0)
#filter.to_csv("filtered_df.csv", sep=';')
data_procs=pd.read_csv("data_process.csv",sep=";")
result = pd.merge(filter, data_procs, how="left", on=["Unnamed: 0"])


irony = "label"
SublabelRed=RedditData[irony]
X_train, X_test, y_train, y_test = train_test_split( SubDataRed, SublabelRed, test_size=0.4, random_state=0)
clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
clf.score(X_test, y_test)
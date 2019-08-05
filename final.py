# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 14:33:33 2019

@author: qxy09
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import os
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#1.分割train，validation，test集--sklearn 读取tran集
#提取单个作家集的train set和 test set
def get_data(file):
    data = pd.read_csv("DCLSA/literature_author_dataset/"+file)
    data.drop(['Unnamed: 0','book'], axis=1,inplace=True)

    x,y = data.ix[:,1],data.ix[:,0]
    #测试集为25%，训练集为75%
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)

    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    print("train_len:"+len(train_x))

    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    print("test_len:"+len(test_x))

    return train_x,train_y,test_x,test_y

#提取作家合集的train和test集
def get_data_all():
    data = pd.read_csv("all_ohc.csv")
    data.drop(['Unnamed: 0','book','author'], axis=1,inplace=True)
    #print(len(data))

    x,y = data.ix[:,1:],data.ix[:,0]
    #测试集为25%，训练集为75%
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    print(len(train_x))

    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    print(len(test_x))

    return train_x,train_y,test_x,test_y

def get_data_social():
    data = pd.read_csv("pan13-author-profiling-training-corpus-2013-01-09\data1.csv", lineterminator='\n')
    #data = pd.read_csv("test_data.csv", lineterminator='\n')
    data.drop(['Unnamed: 0','Unnamed: 0.1','gender'], axis=1,inplace=True)
    #print(len(data))

    x,y = data.ix[:,0],data.ix[:,1]
    #测试集为25%，训练集为75%
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25,random_state=0)
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    print(len(train_x))

    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    print(len(test_x))
    
    train_list=[]
    for i in list(train_x):
        s = i.strip()
        i = s
        train_list.append(i)
    
    test_list=[]
    for i in list(test_x):
        s = i.strip()
        i = s
        test_list.append(i)
    
    train_x1 = pd.Series(train_list)
    test_x1 = pd.Series(test_list)
        
    return train_x1,train_y,test_x1,test_y


#特征提取：word分词，stem，pos，pos+word。---都可以以jason形式保存，随时提取（保存好） 1）先分词，生成list 2）用nltk讲word list变成stem，pos，（pos，word）形式 3）保存各自的list就是该feature
#分词
def word_feature(text):
    word_list = nltk.word_tokenize(text)
    word_list = list(set(word_list))
    return word_list

#feature1:生成lemma
def lemma_feature(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_list = word_feature(text)
    lemma_list = []
    for word in word_list:
        lemma = wordnet_lemmatizer.lemmatize(word)
        lemma_list.append(lemma)

    lemma_list = list(set(lemma_list))    
    return lemma_list

#feature2：生成pos
def pos_feature(text):
    word_list = word_feature(text)
    pos_list = []
    lexical_list = nltk.pos_tag(word_list)
    for pos in lexical_list:
        pos_list.append(pos[1])    
    return pos_list

#feature3:生成lexical（word+pos）
def lexical_feature(text):
    lemma_list = word_feature(text)   
    lexical_list = nltk.pos_tag(lemma_list)
    lexical_list1 = []
    for i in lexical_list:
        temp = i[0]+'.'+i[1]
        lexical_list1.append(temp)
    return lexical_list1

##提取train 集特征值
#1.训练集 lemma特征值
def lemma_f(train_x):
    lemma_f = []
    for text in train_x:
        temp_lemma_list = lemma_feature(text)
        lemma_f = list(set(lemma_f).union(set(temp_lemma_list)))
    return lemma_f
     
#2.训练集 pos特征值
def pos_f(train_x):
    pos_f = []
    for text in train_x:
        temp_pos_list = pos_feature(text)
        pos_f = list(set(pos_f).union(set(temp_pos_list)))
    return pos_f


#3.训练集 lexical特征值
def lexical_f(train_x):
    lexical_f = []
    for text in train_x:
        temp_lexical_list = lexical_feature(text)
        lexical_f = list(set(lexical_f).union(set(temp_lexical_list)))
    return lexical_f     



#向量化生成document-feature matrix
#向量化函数
def vectorize(sent,feature_list):
    vector = [sent.count(f) for f in feature_list]
    return(vector)
    
#22个作家：train和test set 的矩阵生成
def create_matrix(x,y,feature_fun,feature_type):  
    origin_list = x
    feature_list = list(map(feature_fun,origin_list))
    
    vectors = [vectorize(s,feature_type) for s in feature_list]
    df = pd.DataFrame(vectors, columns=feature_type)
    
    df['YEAR'] = y
    print(df['YEAR'])
    return df

#作家全集的矩阵生成
def create_matrix_all(x,y,feature_fun,feature_type):

    #文本变量向量化
    test_list = x['text']
    feature_list = list(map(feature_fun,test_list))
    
    vectors = [vectorize(s,feature_type) for s in feature_list]
    df1 = pd.DataFrame(vectors, columns=feature_type)
    df2 = x.iloc[:,1:]
    
    df = pd.concat([df1,df2],axis=1)
    df['YEAR'] = y
    return df

#social集的矩阵生成
def create_matrix_social(x,y,feature_fun,feature_type):  
    origin_list = x
    feature_list = list(map(feature_fun,origin_list))
    
    vectors = [vectorize(s,feature_type) for s in feature_list]
    df = pd.DataFrame(vectors, columns=feature_type)
    
    df['AGE'] = y
    #print(df['AGE'])
    return df
    

if __name__ == '__main__':
    #生成训练集和测试集
    
    #生成单个作家集的训练集和测试集
    #ab_train_x,ab_train_y,ab_test_x,ab_test_y = get_data("ab.csv")
    
    #生成作家合集的训练集和测试集
    all_train_x,all_train_y,all_test_x,all_test_y = get_data_all()
    
    #生成social media集的训练集和测试集
    social_train_x,social_train_y,social_test_x,social_test_y = get_data_social()
    print(social_train_x.head())
    
    
    #生成对应特征集
    #生成22个作家的三种特征集
    #ab_lemma_f = lemma_f(ab_train_x)
    #ab_pos_f = pos_f(ab_train_x)
    #ab_lexical_f = lexical_f(ab_train_x)
    
    #生成作家合集的三种特征集
    #all_lemma_f = lemma_f(all_train_x['text'])
    #all_pos_f = pos_f(all_train_x['text'])
    all_lexical_f = lexical_f(all_train_x['text'])
    
    #生成social集的三种特征集
    social_lemma_f = lemma_f(social_train_x)
    social_pos_f = pos_f(social_train_x)
    social_lexical_f = lexical_f(social_train_x)
    
    #生成22个作家的矩阵
#    create_matrix(ab_train_x,ab_train_y,lemma_feature,ab_lemma_f).to_csv("DCLSA/la_dataset_split/ab/ab_lemma_train_matrix.csv")
#    create_matrix(ab_test_x,ab_test_y,lemma_feature,ab_lemma_f).to_csv("DCLSA/la_dataset_split/ab/ab_lemma_test_matrix.csv")
#    create_matrix(ab_train_x,ab_train_y,pos_feature,ab_pos_f).to_csv("DCLSA/la_dataset_split/ab/ab_pos_train_matrix.csv")
#    create_matrix(ab_test_x,ab_test_y,pos_feature,ab_pos_f).to_csv("DCLSA/la_dataset_split/ab/ab_pos_test_matrix.csv")
#    create_matrix(ab_train_x,ab_train_y,lexical_feature,ab_lexical_f).to_csv("DCLSA/la_dataset_split/ab/ab_lexical_train_matrix.csv")
#    create_matrix(ab_test_x,ab_test_y,lexical_feature,ab_lexical_f).to_csv("DCLSA/la_dataset_split/ab/ab_lexcial_test_matrix.csv")
    
    #生成作家全集的矩阵
    #create_matrix_all(all_train_x,all_train_y,lemma_feature,all_lemma_f).to_csv("DCLSA/la_dataset_split/all_lemma_train_matrix.csv")
    #create_matrix_all(all_test_x,all_test_y,lemma_feature,all_lemma_f).to_csv("DCLSA/la_dataset_split/all_lemma_test_matrix.csv")
    #create_matrix_all(all_train_x,all_train_y,pos_feature,all_pos_f).to_csv("DCLSA/la_dataset_split/all_pos_train_matrix.csv")
    #create_matrix_all(all_test_x,all_test_y,pos_feature,all_pos_f).to_csv("DCLSA/la_dataset_split/all_pos_test_matrix.csv")
    create_matrix_all(all_train_x,all_train_y,lexical_feature,all_lexical_f).to_csv("DCLSA/la_dataset_split/all_lexical_train_matrix.csv")
    create_matrix_all(all_test_x,all_test_y,lexical_feature,all_lexical_f).to_csv("DCLSA/la_dataset_split/all_lexcial_test_matrix.csv")
    
    #生成social集的矩阵
    create_matrix(social_train_x,social_train_y,lemma_feature,social_lemma_f).to_csv("pan13-author-profiling-training-corpus-2013-01-09/social_lemma_train_matrix.csv")
    create_matrix(social_test_x,social_test_y,lemma_feature,social_lemma_f).to_csv("pan13-author-profiling-training-corpus-2013-01-09/social_lemma_test_matrix.csv")
    create_matrix(social_train_x,social_train_y,pos_feature,social_pos_f).to_csv("pan13-author-profiling-training-corpus-2013-01-09/social_pos_train_matrix.csv")
    create_matrix(social_test_x,social_test_y,pos_feature,social_pos_f).to_csv("pan13-author-profiling-training-corpus-2013-01-09/social_pos_test_matrix.csv")
    create_matrix(social_train_x,social_train_y,lexical_feature,social_lexical_f).to_csv("pan13-author-profiling-training-corpus-2013-01-09/social_lexical_train_matrix.csv")
    create_matrix(social_test_x,social_test_y,lexical_feature,social_lexical_f).to_csv("pan13-author-profiling-training-corpus-2013-01-09/social_lexcial_test_matrix.csv")
    
    
import numpy as np
import pandas as pd
import os
import math
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from bs4 import BeautifulSoup
from scipy.spatial.distance import euclidean, braycurtis, canberra, chebyshev, minkowski
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier,RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def fun(path,i):
        
    files = os.listdir(path)
    os.chdir(path)
    n = len(files)

    for f in files:
        a = open(f).read()
        soup = BeautifulSoup(a, 'html.parser')
        docs.append(soup.get_text())
        class_label.append(i)
removedigit = str.maketrans('', '', string.digits)

def replace_(a):

    s = str.replace(a, ",", " ")
    s = str.replace(s,string.punctuation, " ")
    s = str.replace(s,"(", " ")
    s = str.replace(s, ")", " ")
    s = str.replace(s, '"', " ")
    s = str.replace(s, "'", " ")
    s = str.replace(s,"-"," ")
    s = str.replace(s,"_", "")
    s = s.translate(removedigit)
    return s.lower()

#Similarity Measures

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1,vector2.T)
    magnitude = math.sqrt(np.sum(vector1**2)) * math.sqrt(np.sum(vector2**2))
    if not magnitude:
        return 0
    return dot_product/magnitude

#boolean version
def jaccard_similarity(query, document):
    a = (query>0)
    b = (document>0)
    c = a&b
    d = a|b
    if np.sum(d)==0:
        return 0
    return float(np.sum(c))/float(np.sum(d))

def manhattan_distance(x,y):
    a = abs(x-y)
    s = np.sum(a)
   
    return float(s)

def dice(a,b):
    s = np.dot(a,b.T)
    a_mod = np.sum(a**2)
    b_mod = np.sum(b**2)
    if (a_mod+ b_mod)==0:
        return 0
    return 2. * s/ (a_mod+ b_mod)

def pearsonr(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_sq = np.sum(x**2)
    sum_y_sq = np.sum(y**2)
    psum = np.sum(x*y)
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

def E_jaccard(a,b):
    s = np.dot(a,b.T)
    a2=np.sum(a**2)
    b2=np.sum(b**2)
    if (a2 + b2-s)==0:
        return 0
    return  1.0*s/ (a2 + b2-s)


def hamming(a,b):
    im1 = np.asarray(a).astype(np.bool)
    im2 = np.asarray(b).astype(np.bool)
    s = np.sum(im1!=im2)
    return s

def overlap(a,b):
    im1 = np.asarray(a)
    im2 = np.asarray(b)
    s = np.dot(im1, im2.T)
    a1 = np.sum(im1**2)
    b1 = np.sum(im2**2)
    if min(a1,b1)==0:
        return 0
    return s/min(a1,b1)

i=1
docs=[]
class_label=[]
#update path here
for di in os.listdir('/home/aw/Desktop/plagiarism-teller/train'):
    if len(di)==4:
        path = '/home/aw/Desktop/plagiarism-teller/train/'+di+'/docs'
        fun(path,i)
        i+=1

doc_frame = pd.DataFrame()
doc_frame['docs'] = docs
doc_frame['label'] = class_label
doc_frame['docs'] = doc_frame['docs'].apply(replace_)

docs=[]
class_label=[]
i=1
#update path here
for di in os.listdir('/home/aw/Desktop/plagiarism-teller/test/docs'):
    if len(di)==4:
        path = '//test/docs/'+di
        fun(path,i)
        i+=1


doc_frame_test = pd.DataFrame()
doc_frame_test['docs'] = docs
doc_frame_test['label'] = class_label
doc_frame_test['docs'] = doc_frame_test['docs'].apply(replace_)


english_stemmer = SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


count = StemmedCountVectorizer(min_df=0, stop_words='english', analyzer='word')
doc_frame['doc_vec_count'] = list(count.fit_transform(doc_frame['docs']).toarray())

vec_size = len(doc_frame['doc_vec_count'][0])
cij = np.zeros((len(doc_frame), vec_size))

for i in range(len(doc_frame)):
    cij[i] = doc_frame['doc_vec_count'][i]

a = np.matmul(cij.T,cij)

for i in range(vec_size):
    for j in range(vec_size):
         if(j>i):
            a[i][j] = a[i][j]/(a[i][i]+a[j][j]-a[i][j])

for i in range(vec_size):
    for j in range(vec_size):
        a[j][i]=a[i][j]
    a[i][i]=1
sc = np.zeros(vec_size)

for i in range(vec_size):
    sc[i] = np.sum(a[i])/vec_size


word_similarity = []
for i in range(vec_size):
    cossim = cosine_similarity(a[i],sc)
    word_similarity.append(cossim)

words = count.get_feature_names()

features = pd.DataFrame()

features['words'] = words

features['word_similarity'] = word_similarity

sor = features.sort_values('word_similarity', ascending=False)

feature_list = ['cosine_similarity', 'euclidean_distance', 'E_jaccard', 'Pearson_Correlation', 'dice', 'manhattan_distance','jaccard_similarity', 'Bray_Curtis_Distance', 'Canberra_Distance','Chebyshev Distance', 'hamming', 'overlap', 'Minkowski Distance']

for th in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    voc = list(sor[sor['word_similarity']>=th]['words'])
    print("Threshold: ", th, "No of words", len(voc))
    count = StemmedCountVectorizer(min_df=0, stop_words='english', analyzer='word', vocabulary=voc)
    doc_frame['doc_vec_count'] = list(count.fit_transform(doc_frame['docs']).toarray())
    filter_docs = []
    filter_label = []
    for i in range(len(doc_frame['doc_vec_count'])):
        if(doc_frame['doc_vec_count'][i].sum() > 0):#checking if there area any docs which have no words from top k words.
            filter_docs.append(doc_frame['docs'][i])
            filter_label.append(doc_frame['label'][i])
    print("No of docs in filtered training data: ", len(filter_docs))
    filter_frame = pd.DataFrame()
    filter_frame['docs'] = filter_docs
    filter_frame['label'] = filter_label
    tfidf = StemmedTfidfVectorizer(min_df=0, stop_words='english', analyzer='word', vocabulary=voc)
    filter_frame['doc_vec'] = list(tfidf.fit_transform(filter_frame['docs']).toarray()) 
    print("Length of word vector: ", len(filter_frame['doc_vec'][0]))
    
    
    cs =[]
    doc_ids = []
    same_class = []
    jc =[]
    euclid = []  
    man = []
    dc=[]
    cor=[]
    e_jac=[]
    bray_cur = []
    canbe= []
    cheby = []
    ham=[]
    olp=[]
    mink=[]
    print("Computing Similarity Measures on Filtered Training Data")
    for i in range(len(filter_frame)-1):
        for j in range(i+1,len(filter_frame)):
            doc_ids.append((i,j))
            if filter_frame['label'][i]==filter_frame['label'][j]:
                same_class.append(1)
            else:
                same_class.append(0)
            a = cosine_similarity(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            cs.append(a)     
            a = jaccard_similarity(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            jc.append(a)
            a = euclidean(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            euclid.append(a)
            a = manhattan_distance(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            man.append(a)
            a = dice(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            dc.append(a)
            a = pearsonr(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            cor.append(a)
            a = E_jaccard(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            e_jac.append(a)
            a = braycurtis(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            bray_cur.append(a)
            a = canberra(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            canbe.append(a)
            a = chebyshev(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            cheby.append(a)
            a = hamming(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            ham.append(a)
            a = overlap(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j])
            olp.append(a)
            a = minkowski(filter_frame['doc_vec'][i], filter_frame['doc_vec'][j], 3)
            mink.append(a)



    print("Finished Computing Similarity Measures on Filtered Training Data")

    doc_pair = pd.DataFrame()
    doc_pair['ids'] = doc_ids
    doc_pair['Same_Class'] = same_class
    doc_pair['cosine_similarity'] = cs
    doc_pair['euclidean_distance'] = euclid
    doc_pair['E_jaccard'] = e_jac
    doc_pair['Pearson_Correlation'] = cor
    doc_pair['dice'] = dc
    doc_pair['manhattan_distance']= man
    doc_pair['jaccard_similarity'] = jc
    doc_pair['Bray_Curtis_Distance'] = bray_cur
    doc_pair['Canberra_Distance'] = canbe
    doc_pair['Chebyshev Distance'] = cheby
    doc_pair['hamming'] = ham
    doc_pair['overlap'] = olp
    doc_pair['Minkowski Distance'] = mink

    print("Creating SVM Classifier with Linear Kernel on training Data")
    svlk=svm.SVC(kernel='linear')
    svlk.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("SVM Classifier with Linear Kernel on training Data Created")
    
    print("Creating LinearSVC Classifier on training Data")
    lsvc=svm.LinearSVC(random_state=0)
    lsvc.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("LinearSVC Classifier on training Data Created")
    
    print("Creating KNN Classifier on training Data")
    knn=neighbors.KNeighborsClassifier()
    knn.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("KNN Classifier on training Data Created")
    
    print("Creating Gaussian NB Classifier on training Data")
    gnb = GaussianNB()
    gnb.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("GaussianNB Classifier on training Data Created")
    
    
    print("Creating Random Forest Classifier on training Data")
    rf = RandomForestClassifier(max_depth=3, random_state=0)
    rf.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("Random Forest Classifier on training Data Created")

    print("Creating Decision Trees CLassifier on training Data")
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("Decision Trees CLassifier on training Data Created")
    
    print("Creating Ada boost Classifier on training Data")
    ab=  AdaBoostClassifier()
    ab.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("Adaboost Classifier on training Data Created")
    
    print("Creating Gradient Boosting Classifier on training Data")
    gbc=GradientBoostingClassifier()
    gbc.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("Gradient Boosting Classifier on training Data Created")
 
    print("Creating Extra Trees Classifier on training Data")
    etc=  ExtraTreesClassifier()
    etc.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("Extra Trees Classifier on training Data Created")
    
    print("Creating MLPC Classifier on training Data")
    mlpc=  MLPClassifier(alpha=0.01)
    mlpc.fit(doc_pair[feature_list], doc_pair['Same_Class'])
    print("MLPC Classifier on training Data Created")
    
    
    doc_frame_test['doc_vec'] = list(tfidf.transform(doc_frame_test['docs']).toarray())
    cs =[]
    doc_ids = []
    same_class = []
    jc =[]
    euclid = []  
    man = []
    dc=[]
    cor=[]
    e_jac=[]
    bray_cur = []
    canbe= []
    cheby = []
    ham=[]
    olp=[]
    mink=[]
    print("Computing Similarity Measures on Test Data Data")
    for i in range(len(doc_frame_test)-1):
        for j in range(i+1,len(doc_frame_test)):
            doc_ids.append((i,j))
            a = cosine_similarity(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            cs.append(a)
            if doc_frame_test['label'][i]==doc_frame_test['label'][j]:
                same_class.append(1)
            else:
                same_class.append(0)
            a = jaccard_similarity(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            jc.append(a)
            a = euclidean(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            euclid.append(a)
            a = manhattan_distance(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            man.append(a)
            a = dice(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            dc.append(a)
            a = pearsonr(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            cor.append(a)
            a = E_jaccard(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            e_jac.append(a)
            a = braycurtis(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            bray_cur.append(a)
            a = canberra(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            canbe.append(a)
            a = chebyshev(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            cheby.append(a)
            a = hamming(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            ham.append(a)
            a = overlap(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j])
            olp.append(a)
            a = minkowski(doc_frame_test['doc_vec'][i], doc_frame_test['doc_vec'][j], 3)
            mink.append(a)
    doc_pair_test = pd.DataFrame()
    doc_pair_test['ids'] = doc_ids
    doc_pair_test['Same_Class'] = same_class
    doc_pair_test['cosine_similarity'] = cs
    doc_pair_test['euclidean_distance'] = euclid
    doc_pair_test['E_jaccard'] = e_jac
    doc_pair_test['Pearson_Correlation'] = cor
    doc_pair_test['dice'] = dc
    doc_pair_test['manhattan_distance']= man
    doc_pair_test['jaccard_similarity'] = jc
    doc_pair_test['Bray_Curtis_Distance'] = bray_cur
    doc_pair_test['Canberra_Distance'] = canbe
    doc_pair_test['Chebyshev Distance'] = cheby
    doc_pair_test['hamming'] = ham
    doc_pair_test['overlap'] = olp
    doc_pair_test['Minkowski Distance'] = mink
    print("Finished Computing Similarity Measures on Test Data")
    
    doc_pair_test['Predicted'] = knn.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("KNN Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = gnb.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("Gaussian nb Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = rf.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("R forest Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)

    doc_pair_test['Predicted'] = dt.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("D trees Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)

    doc_pair_test['Predicted'] = ab.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("ada boost Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = svlk.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("SVM Linear Kernel Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = lsvc.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("Linear SVC Model")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = gbc.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("Gradient Boosting Classifier")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = etc.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("Extra Tree Classifier")
    print("Threshold: ",th," Accuracy: ", acc, " F1-Score:  ", f1score)
    
    doc_pair_test['Predicted'] = mlpc.predict(doc_pair_test[feature_list])
    acc=accuracy_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    f1score = f1_score(doc_pair_test['Same_Class'], doc_pair_test['Predicted'])
    print("MLPC Classifier")
    print("Threshold: ",th, " Accuracy: ", acc, " F1-Score:  ", f1score)

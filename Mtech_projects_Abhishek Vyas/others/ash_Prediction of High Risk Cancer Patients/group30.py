import argparse
import warnings
from collections  import defaultdict
import  pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler,MinMaxScaler,normalize
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import matthews_corrcoef,accuracy_score
#!pip install tensorflow-addons
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
import tensorflow as tf
tf.random.Generator = None
import keras as k
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import joblib
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import tensorflow_addons as tfa
from tensorflow.keras import datasets,layers,models
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout
import tensorflow as tf
from sklearn.model_selection import KFold
import keras as k
from tensorflow.keras import datasets,layers,models
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Please provide following arguments to proceed')

## Read Arguments from commandline
parser.add_argument("-tr", "--training", type=str, required=True, help="Input: Provide input training file location folder")
parser.add_argument("-te", "--testing", type=str, required=True,help="Input: Provide input testing file location folder")
parser.add_argument("-o","--output",type=str, help="Enter the output/submission file name")
# Parameter initialization
args = parser.parse_args()
if args.output == None:
    out= "outfile.csv"
else:
    out= args.output
tr=args.training
te=args.testing
# Reading the input data
pic='m1'
f_t=tr
data=pd.read_csv(f_t)
y=data['Labels']
x=data.drop(columns=['Labels','ID'])

# Normalizing or standardizing the train data using StandardScakar/ MinMax Scaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
x1 = pd.DataFrame(scaler.fit_transform(x))
x.columns=x1.columns
# Splitting unpadded data into train and test
x_train_up,x_test_up,y_train_up,y_test_up=train_test_split(x,y,test_size=0.2,random_state=0)
# Appending extra columns to make data idle for applying cnn
for i in range(6):
  x['pd_'+str(i)]=[0 for i in range(len(y))]
# Splitting padded data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_tr_mpl=x_train
x_te_mpl=x_test
y_tr_mpl=y_train
y_te_mpl=y_test
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

# reshaping the data
x_train=np.reshape(x_train, (x_train.shape[0],18,18,1))
x_test=np.reshape(x_test, (x_test.shape[0],18,18,1))

#CNN with 5 fold Cross validation
folds = KFold(n_splits=5, shuffle=True)
mcc1=[]
loss= []
xi = np.concatenate((x_train, x_test), axis=0)
yi = np.concatenate((y_train, y_test), axis=0)

# For each fold
for tra, tes in folds.split(xi, yi):
  model_cv = models.Sequential()
  model_cv.add(layers.Conv2D(250, (3, 3), input_shape=(18,18,1)))
  model_cv.add(layers.Flatten())
  model_cv.add(layers.Dense(1,activation='sigmoid'))
  mcc=tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
  model_cv.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mcc])
  model_cv.fit(xi[tra], yi[tra], epochs=10)
  scores = model_cv.evaluate(xi[tes], yi[tes], verbose=0)
  mcc1.append(scores[1])
  loss.append(scores[0])
# Calling the sequential Model with 3 layers Conv2D, Flatten and Dense layer
model = models.Sequential()
model.add(layers.Conv2D(250, (3, 3), input_shape=(18,18,1)))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))
print("************* Model summary *************")
print(model.summary())

# using tensorflow addons mcc metric is imported with num_classes=1
mcc=tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)

# Sequential model is compiled with adam optimizer and binary_crossentropy as loss function and mcc as a evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mcc])
# Model is then fitted on training data with 20% validation data and 100 epochs
model.fit(x_train, y_train, epochs=100,validation_data=(x_test, y_test))
# validation data label prediction is done using trained model
y_pred=model.predict(x_test)
y_t=[]
for i in y_pred:
  y_t.append(int(i))
# print(matthews_corrcoef(y_t,y_test))
# print(accuracy_score(y_t,y_test))
# print(y_t)
# a=[y_t,y_test]
# import joblib
# with open('drive/My Drive/mlba1/m1','wb') as fo:
#   joblib.dump(a,fo)
# # a=[y_t,y_test]

with open(pic,'rb') as fo:
  d=joblib.load(fo)
y_t=d[0]
y_test=d[1]

f2=te

# testing data is read and same preprocessing is done as training data
data2=pd.read_csv(f2)
test_id=data2['ID']
test_data=data2.drop(columns=['ID'])
test_data_up=data2.drop(columns=['ID'])
test_data_up1=data2.drop(columns=['ID'])
test_data_up2=data2.drop(columns=['ID'])
# scaler = StandardScaler()
scaler = MinMaxScaler()
x2 = pd.DataFrame(scaler.fit_transform(test_data))
x3 = pd.DataFrame(scaler.fit_transform(test_data_up))
x4 = pd.DataFrame(scaler.fit_transform(test_data_up1))
x5 = pd.DataFrame(scaler.fit_transform(test_data_up2))
test_data.columns=x2.columns
test_data_up1.columns=x4.columns
test_data_up2.columns=x5.columns
test_data_up.columns=x3.columns
for i in range(6):
  test_data['pd_'+str(i)]=[0 for i in range(len(test_data))]
for i in range(6):
  test_data_up1['pd_'+str(i)]=[0 for i in range(len(test_data_up1))]
test_data=np.array(test_data)
test_data_up1=np.array(test_data_up1)
test_data=np.reshape(test_data,(test_data.shape[0],18,18,1))

# then test data labels are predicted with same model
y_p_t=model.predict(test_data)
y_p_t1=[]
for i in y_p_t:
  y_p_t1.append(int(i))

# creating a submission file
y1=pd.DataFrame()
y1['ID']=test_id
y1['Labels']=y_p_t1
y1.to_csv(out,index=False)

# creating ANN model with mcc metric
model1 = keras.Sequential([
    keras.layers.Dense(1, input_shape=(x_train_up.shape[1],), activation='sigmoid')
])
mcc=tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1)
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=[mcc])
model1.fit(x_train_up,y_train_up, epochs=100)
# predicting validation labels
y_pred_ann=model1.predict(x_test_up)
y_t_ann=[]
for i in y_pred_ann:
  y_t_ann.append(int(i))
# then test data labels are predicted with same model
y_p_t2=model1.predict(test_data_up)
y_p_t1=[]
for i in y_p_t2:
  y_p_t1.append(int(i))

# creating a submission file
y1=pd.DataFrame()
y1['ID']=test_id
y1['Labels']=y_p_t1
o1=out.split('.')
out1=o1[0]+'1.'+o1[1]
y1.to_csv(out1,index=False)



print("\n")
print("************************************************ CNN *******************************************************")
print("MCC value for validation data using CNN is:- ",matthews_corrcoef(y_t,y_test))
print("Accuracy value for validation data using CNN is:- ",accuracy_score(y_t,y_test))
print("\n")
print("*********************************** CNN with 5 fold Cross Validation ***************************************")
print("5 fold mean test MCC is:- ",np.mean(mcc1))
print("5 fold mean test Binary_crossentropy Loss is:- ", np.mean(loss))
print("\n")
print("*********************************Artificial Neural Network *************************************************")
print("MCC Score According to Artificial neural network is:- ",matthews_corrcoef(y_t_ann,y_test_up))
print("Accuracy Score According to Artificial neural network is:- ",accuracy_score(y_t_ann,y_test_up))
print("\n")
print("*********************************** Neural Network (MLPClassifier) *****************************************")
# Neural Network implementation using sklearn MLPClassifier.
clf = MLPClassifier()
clf.fit(x_tr_mpl,y_tr_mpl)
mlp_pred=clf.predict(x_te_mpl)
print("MCC Score According to neural network is:- ",matthews_corrcoef(mlp_pred,y_te_mpl))
print("Accuracy Score According to neural network is:- ",accuracy_score(mlp_pred,y_te_mpl))
# then test data labels are predicted with same model
y_p_t3=clf.predict(test_data_up1)
y_p_t1=[]
for i in y_p_t3:
  y_p_t1.append(int(i))

# creating a submission file
y1=pd.DataFrame()
y1['ID']=test_id
y1['Labels']=y_p_t1
o2=out.split('.')
out2=o2[0]+'2.'+o2[1]
y1.to_csv(out2,index=False)
print("____________________________________________________________________________________________________________")
# Neural Network implementation using sklearn MLPClassifier without padding the data
clf1 = MLPClassifier(alpha=0.1, hidden_layer_sizes=10, max_iter=15, random_state= 9, solver='lbfgs')
clf1.fit(x_train_up,y_train_up)
mlp_pred1=clf1.predict(x_test_up)
y_p_t3=clf1.predict(test_data_up2)
y_p_t1=[]
for i in y_p_t3:
  y_p_t1.append(int(i))

# creating a submission file
y1=pd.DataFrame()
y1['ID']=test_id
y1['Labels']=y_p_t1
o3=out.split('.')
out3=o3[0]+'3.'+o3[1]
y1.to_csv(out3,index=False)
print("MCC Score(unpadded) According to neural network is:- ",matthews_corrcoef(mlp_pred1,y_test_up))
print("Accuracy Score(unpadded) According to neural network is:- ",accuracy_score(mlp_pred1,y_test_up))
print("____________________________________________________________________________________________________________")
# Neural Network implementation using sklearn MLPClassifier with feature selection using VarianceThreshold
sel = VarianceThreshold(1.0)
sf = sel.fit_transform(x_train_up)
sc = x_train_up.columns[sel.get_support(indices=True)]
x_train_up[sc].head()

# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train_up, y_train)
# model = SelectFromModel(lsvc, prefit=True)
# sf = model.transform(x_train_up_train)
# ab = x_train_up.columns[model.get_support(indices=True)]
# x_train_up[ab].head()

clf2 = MLPClassifier(random_state=1, max_iter=300).fit(x_train_up[sc], y_train_up)
scores = cross_val_score(clf2, x_train_up[sc], y_train_up, cv=5, scoring='f1_macro')
scores1 = cross_val_score(clf2, x_train_up[sc], y_train_up, cv=5)
y_pred_2 = clf2.predict(x_test_up[sc])
y_p_t4=clf2.predict(test_data_up2[sc])
y_p_t1=[]
for i in y_p_t4:
  y_p_t1.append(int(i))

# creating a submission file
y1=pd.DataFrame()
y1['ID']=test_id
y1['Labels']=y_p_t1
o4=out.split('.')
out4=o4[0]+'4.'+o4[1]
y1.to_csv(out4,index=False)
print("MCC Score(unpadded) according to neural network with setting variance threshold as 1 is:- ",matthews_corrcoef(y_test_up, y_pred_2))
print("Accuracy Score(unpadded) according to neural network with setting variance threshold as 1 is:- ",accuracy_score(y_test_up, y_pred_2))
print("Mean Accuracy of 5 fold cross validation: ",scores.mean())
print("MCC of 5 fold cross validation: ",scores1.mean())
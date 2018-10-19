import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

default=pd.read_csv('default.csv',index_col="ID")
default.rename(columns=lambda x: x.lower(),inplace=True)

default['grad_school']=(default['education']==1).astype('int')
default['university']=(default['education']==2).astype('int')
default['high_school']=(default['education']==3).astype('int')

default.drop('education',axis=1,inplace=True)

default['male']=(default['sex']==1).astype('int')
default.drop('sex',axis=1,inplace=True)

default['married']=(default['marriage']==1).astype('int')
default.drop('marriage',axis=1,inplace=True)

pay_features=['pay_0','pay_2','pay_3','pay_4','pay_6']
for p in pay_features:
    default.loc[default[p]<=0,p]=0

default.rename(columns={'default_payment': 'default'},inplace=True)




#default.to_csv('out.csv', sep=',')


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
from sklearn.preprocessing import RobustScaler

target_name='default'
X=default.drop('default',axis=1)
robust_scaler=RobustScaler()
X=robust_scaler.fit_transform(X)
y=default[target_name]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=123,stratify=y)

def CMatrix(CM,labels=['NOT-DELAY','DELAY ']):
    df=pd.DataFrame(data=CM,index=labels,columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total']=df.sum()
    df['Total']=df.sum(axis=1)
    return df


metrics=pd.DataFrame(index=['accuracy','precision','recall'],
                     columns=['NULL','LogisticReg','ClassTree','NaiveBayes'])


y_pred_test=np.repeat(y_train.value_counts().idxmax(),y_test.size)
metrics.loc['accuracy','NULL']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','NULL']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','NULL']=recall_score(y_pred=y_pred_test,y_true=y_test)
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)

print("MODEL0")
print("Confusion Matrix for NULL MODEL")
print("--------------------------------------------------")

print(CMatrix(CM))
print("--------------------------------------------------")
print("MODEL1")
print("Confusion Matrix for LOGISTICREGRESSION MODEL")
print("--------------------------------------------------")


#LogisticRegression Model

from sklearn.linear_model import LogisticRegression
logistic_regression=LogisticRegression(n_jobs=-1,random_state=15)
logistic_regression.fit(X_train,y_train)
y_pred_test=logistic_regression.predict(X_test)
metrics.loc['accuracy','LogisticReg']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','LogisticReg']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','LogisticReg']=recall_score(y_pred=y_pred_test,y_true=y_test)
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print(CMatrix(CM))



print("--------------------------------------------------")
print("#MODEL2")
print("Confusion Matrix for ClassTreeClassifier MODEL")
print("--------------------------------------------------")



#DecisionTree Model

from sklearn.tree import DecisionTreeClassifier
class_tree=DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10,random_state=10)
class_tree.fit(X_train,y_train)
y_pred_test=class_tree.predict(X_test)
metrics.loc['accuracy','ClassTree']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','ClassTree']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','ClassTree']=recall_score(y_pred=y_pred_test,y_true=y_test)
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print(CMatrix(CM))



print("--------------------------------------------------")
print("#MODEL3")
print("Confusion Matrix for NaiveBayes Classifier MODEL")
print("--------------------------------------------------")


#NaiveBayes Classifier Model

from sklearn.naive_bayes import GaussianNB
NBC=GaussianNB()
NBC.fit(X_train,y_train)
y_pred_test=NBC.predict(X_test)
metrics.loc['accuracy','NaiveBayes']=accuracy_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['precision','NaiveBayes']=precision_score(y_pred=y_pred_test,y_true=y_test)
metrics.loc['recall','NaiveBayes']=recall_score(y_pred=y_pred_test,y_true=y_test)
CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print(CMatrix(CM))

print("------------------------------------------------------------")
print("comparsion Between LogisticReg,ClassTree,NaiveBayes Model")
print("------------------------------------------------------------")

print(100*metrics)
print("---------------------------------------------------")


fig,ax=plt.subplots(figsize=(8,5))
metrics.plot(kind='barh',ax=ax)
plt.xlabel('comparsion between LogisticReg,ClassTree,NaiveBayes Models')
plt.show()


precision_nb,recall_nb,thresholds_nb=precision_recall_curve(y_true=y_test,
                                                            probas_pred=NBC.predict_proba(X_test)[:,1])
precision_lr,recall_lr,thresholds_lr=precision_recall_curve(y_true=y_test,
                                                            probas_pred=logistic_regression.predict_proba(X_test)[:,1])



fig,ax=plt.subplots(figsize=(8,5))
ax.plot(precision_nb,recall_nb,label='NaiveBayes')
ax.plot(precision_lr,recall_lr,label='LogisticReg')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precesion-Recall Curve')
ax.hlines(y=0.5,xmin=0,xmax=1,color='red')
ax.legend()
plt.show()



#confusion matrix for modified LogisticRegression Classifier

fig,ax=plt.subplots(figsize=(8,5))
ax.plot(thresholds_lr,precision_lr[1:],label='Precision')
ax.plot(thresholds_lr,recall_lr[1:],label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Precision,Recall')
ax.set_title('Logistic Regression classifier: Precision-Recall')
ax.hlines(y=0.6,xmin=0,xmax=1,color='red')
ax.legend()
plt.show()



#Classifier with threshod of 0.2
print("---------------------------------------------------------------")
print("Classifier with threshold of 0.2")


y_pred_proba=logistic_regression.predict_proba(X_test)[:,1]
y_pred_test=(y_pred_proba>=0.2).astype('int')

CM=confusion_matrix(y_pred=y_pred_test,y_true=y_test)
print("Recall:",100*recall_score(y_pred=y_pred_test,y_true=y_test))
print("Precision:",100*precision_score(y_pred=y_pred_test,y_true=y_test))

print("Confusion Matrix")
print(CMatrix(CM))
print("-------------------------------------------------------------------------------")

#print(default)
#Making Individual Prediction

from sklearn.preprocessing import RobustScaler
robust_scaler=RobustScaler()
def make_ind_prediction(new_data):
    data=new_data.values.reshape(1,-1)
    data=robust_scaler.fit_transform(data)
    prob=logistic_regression.predict_proba(data)[0][1]
    if prob>=0.2:
        return 'Will Delay'
    else:
        return 'Will Not-Delay'


pay=default[default['default']==0]
pay.head()



from collections import OrderedDict


new_customer=OrderedDict([
                          ('limit_bal',90000),
                          ('age',34),
                          ('pay_0',0),
                          ('pay_2',0),
                          ('pay_3',0),
                          ('pay_4',0),
                          ('pay_5',0),
                          ('pay_6',0),
                          ('bill_amt1',29239),
                          ('bill_amt2',14027),
                          ('bill_amt3',13559),
                          ('bill_amt4',14331),
                          ('bill_amt5',14948),
                          ('bill_amt6',15549),
                          ('pay_amt1',1518),
                          ('pay_amt2',1500),
                          ('pay_amt3',1000),
                          ('pay_amt4',1000),
                          ('pay_amt5',1000),
                          ('pay_amt6',5000),
                          ('deptime',2003),
                           ('arrtime',2211),
                           ('flightnum',335),
                           ('airtime',116),
                            ('arrdelay',-14),
                            ('depdelay',8),
                           ('distance',810),
                           ('cancelled',0),
                           ('diverted',0),
                           ('grad_school',0),
                          ('university',1),
                          ('high_school',0),
                          ('male',1),
                          ('married',1)
                          ])

new_customer=pd.Series(new_customer)
print("----------------------------------------------------------------")
print("Prediction of Input data")
print(make_ind_prediction(new_customer))











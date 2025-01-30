if __name__ == "__main__":

    #Importing some libraries
	import numpy as np
	import pandas as pd
	import datetime
	import time
	
from firebase import firebase
fixefixed_interval = 3
#firebase = firebase.FirebaseApplication('https://heart-1314b-default-rtdb.firebaseio.com/', None)
firebase = firebase.FirebaseApplication('https://heart-disease-194eb-default-rtdb.firebaseio.com/', None)
count=1

    
    #Getting rid of pesky warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
np.warnings.filterwarnings('ignore')

column_names = [
            "age",  #1
            "sex",  #2
            "cp",  #3
            "trestbp",  #4
            "chol",  #5
            "fbs",   #6
            "restecg",  #7
            "thalach",  #8
            "exang", #9
            "oldpeak",  #10
            "slope",  #11
            "ca", #12
            "thal", #13
            "target"  #14
        ]
    #Importing the dataset
location = './dataset/Preprocessed_Dataset.csv'
dataset = pd.read_csv(location)
dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.iloc[:,dataset.columns != 'target'].values
y=dataset.iloc[:,dataset.columns=='target'].values
    
    #Replace all 'heart-disease' values greater than 0 because my goal is not to classify the disease type
    #for x,i in enumerate(y):
     #   if i>0:y[x]=1
    #Splitting the dataset into the Training set and Test set
from sklearn.model_selection._split import train_test_split
    #from imblearn.combine import SMOTEENN
    #smote_enn = SMOTEENN()
    #X_resampled, y_resampled = smote_enn.fit_sample(X, Y)
X_train, X_test, Y_Train, Y_Test = train_test_split(X,y, test_size=0.3)

    #Feature scaling
    #from sklearn.preprocessing import StandardScaler
    #sc_X = StandardScaler()
    #X_train = sc_X.fit_transform(X_train)
    #X_test = sc_X.transform(X_test)

    #Using Pipeline
import sklearn.pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
from imblearn.pipeline import make_pipeline


    
    #select = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
clf = MLPClassifier(solver='lbfgs', learning_rate='constant', activation='tanh')
kernel = KernelPCA()
    
pipeline = make_pipeline(kernel, clf)
pipeline.fit(X_train, Y_Train)

    #User-input
print(" \n \n ")
print(" \n \n ")
print("###########################################################################################")
print("##                       Heart attack Possibility prediction                             ##")
print("###########################################################################################")
v = []

for i in column_names[:-1]:
    print("--------------------------------------------------------------------------------------------")
    v.append(input("\t\t\t"+i+"\t\t\t|\t\t"))
    
answer = np.array(v)

answer = answer.reshape(1,-1)
print(answer)
#answer = sc_X.transform(answer)

#print ("Predicts:"+ str(pipeline.predict(answer)))
    #print ("("Predicts: " + str(pipeline.predict(answer))")
#print(type(answer))
x = pipeline.predict(pd.DataFrame(answer))
#print(x)
#x = np.array([(pipeline.predict(answer))])
#print(x)
if x[0]==1:
    print("###########################################################################################")
    print("##                              Please check your phone                                  ##")
    print("###########################################################################################")
    print("predicted as Heart disease")
    datetime1=datetime.datetime.now()
    date=datetime1.strftime("%x")
    time=datetime1.strftime("%X")
    day=datetime1.strftime("%A")
    device = "42"
    status="heart-disease"
    data={"Device_ID":device,"Status":status,"Date":date,"Time":time,"Day":day}
    firebase.put('', 'heart disease/Location 1', data)
    #time.sleep(10)
    '''status="clear"
    data={"Device_ID":device,"Status":status,"Date":date,"Time":time,"Day":day}
    firebase.put('', 'heart disease/Location 1', data)
    count=0'''
		
          
else:
    print("###########################################################################################")
    print("##                                 Please check your phone                               ##")
    print("###########################################################################################")

    print("predicted as not a heart disease")
    datetime1=datetime.datetime.now()
    date=datetime1.strftime("%x")
    time=datetime1.strftime("%X")
    day=datetime1.strftime("%A")
    device = "42"
    status="No heart-disease"
    data={"Device_ID":device,"Status":status,"Date":date,"Time":time,"Day":day}
    firebase.put('', 'heart disease/Location 1', data)
    #time.sleep(10)
    #status="clear"
    #data={"Device ID":device,"Status":status,"Date":date,"Time":time,"Day":day}
    #firebase.put('', 'heart disease/Location 1', data)
    #count=0
			 



		

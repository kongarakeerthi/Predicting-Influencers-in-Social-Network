import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer

# Function for feature scaling and transformation
def feature_transform(fn_name, data):
    #Create scaling or transformation object based on user input
    if fn_name == 'standard':
        tran_fn = StandardScaler()
    elif fn_name =='minmax':
        tran_fn = MinMaxScaler()
    elif fn_name =='log':
        tran_fn = FunctionTransformer(np.log1p, validate=True)
    elif fn_name =='normalize':
        tran_fn = Normalizer()
    
    #Applying transformation
    transfx_data = tran_fn.fit_transform(data.astype(float))
    #Converting back to dataframe 
    transfx_data = pd.DataFrame(transfx_data, columns = data.columns)

    return transfx_data
    

#Loading data
filename = 'train.csv'
data = pd.read_csv(filename)

data.head()

data.info()

#Checking for duplicate entries
data.duplicated().sum()

#Droping duplicate entries
data=data.drop_duplicates()
data.info()

#Seperating dependent and independent variables from data
Y = np.asarray(data['Choice'])
X = data.drop(['Choice'],axis=1)

seed=4 # Setting seed value for reproducibility
# Spliting data into 90% training set and 10% test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)


def get_model():
    models = []
    models.append(('LR' , LogisticRegression(solver='liblinear')))
    #models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB' , GaussianNB()))
    models.append(('SVM', SVC(probability=True, gamma='auto')))
    #models.append(('GBC', GradientBoostingClassifier()))
    #models.append(('RF' , RandomForestClassifier(n_estimators=100)))
    models.append(('MLP', MLPClassifier()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    return models

def base_line_performance(X_train, y_train,models):
    num_folds = 10
    scoring = 'accuracy'
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print('{}: {:.4} ({:.4})'.format(name, cv_results.mean(), cv_results.std()))
        
    return names, results

print("Testing before transform")
models = get_model()
#names,results = base_line_performance(X_train, y_train,models)



#Applying different scaling and log transformation 
#X_train_minmax = feature_transform('minmax',X_train)
#X_train_standard = feature_transform('standard',X_train)
#X_train_norm = feature_transform('normalize',X_train)
X_train_log = feature_transform('log',X_train)

#Checking model performance after minmax scaling
#names,results_minmax = base_line_performance(X_train_minmax, y_train,models)

#Checking model performance after Standard scaling
#names,results_standard = base_line_performance(X_train_standard, y_train,models)

#Checking model performance after Normalization
#names,results_norm = base_line_performance(X_train_norm, y_train,models)

#Checking model performance after Log transformation
names,results_log = base_line_performance(X_train_log, y_train,models)

print("Completed transform")
#Creating a new binary feature based on follower count
##assigning 1 if a person has more than a million follower
data['A_is_popular'] =  [1 if x >= 1000000 else 0 for x in data['A_follower_count']]
data['B_is_popular'] =  [1 if x >= 1000000 else 0 for x in data['B_follower_count']]

#Creating new feature by deviding a follower count from following count
data['A_popularity_score'] = data['A_following_count'].divide(data['A_follower_count'])
data['B_popularity_score'] = data['B_following_count'].divide(data['B_follower_count'])

X = data.drop(['Choice'],axis=1)
print("spliting with new featues")
# Spliting data into 90% training set and 10% test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)

print(" new feture trans form")
#X_train_log = feature_transform('log',X_train)
#names,results_log = base_line_performance(X_train_log, y_train,models)

print("selecting ")
#Checking 6 Least important features using Univariate Selection
from sklearn.feature_selection import SelectKBest, chi2
test = SelectKBest(chi2, k=20)
test.fit_transform(X_train_log, y_train)
index=sorted(range(len(test.scores_)), key=lambda k: test.scores_[k])[0:6]
X_train_log.columns[index]

X_train_log.head()

#Checking 6 Least important features for non-tree based models using Recursive Feature Elemination
#from sklearn.feature_selection import RFE
#estimator = LogisticRegression(solver='liblinear')
#selector = RFE(estimator, 20, step=1)
#selector = selector.fit(X_train_log, y_train)
#index=list(np.where(selector.support_== 0))
#X_train_log.columns[index]

#Checking important feature for tree based models using feature importance attribute 
estimator = GradientBoostingClassifier()
estimator.fit(X_train_log, y_train)
n_features = X_train_log.shape[1]
plt.figure(figsize=(12,8))
plt.barh(range(n_features), estimator.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train_log.columns)
plt.xlabel("Feature importance")
plt.ylim(-1, n_features)

#Removing least important features
cols = ['A_following_count','B_following_count']
data = data.drop(cols,axis=1)

X = data.drop(['Choice'],axis=1)
# Spliting data into 90% training set and 10% test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)

X_train_log = feature_transform('log',X_train)
names,results_log = base_line_performance(X_train_log, y_train,models)



from sklearn.model_selection import GridSearchCV
class GridSearch(object):
    
    def __init__(self,X_train,y_train,model,hyperparameters):
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        
    def GridSearch(self):
        cv = 10
        clf = GridSearchCV(self.model,self.hyperparameters,cv=cv,n_jobs=-1)
        # Fit randomized search
        best_model = clf.fit(self.X_train, self.y_train)
        print("Best: {:.4} using {}".format(best_model.best_score_,best_model.best_params_))
        return best_model,best_model.best_params_
    
    def best_model_pridict(self,X_test):
        best_model, best_param = self.GridSearch()
        pred = best_model.predict(X_test)
        return pred,best_param

# Applying log transform to test data
X_test_log = feature_transform('log',X_test)

#Applying grisearch for losgistic regression model
model_LR = LogisticRegression()
r=np.random.uniform(-2,1,20)
#Hyperparameters to test
param_LR = {'C':10**r}

LR_gridsearch = GridSearch(X_train_log,y_train,model_LR,param_LR)
pred_LR,best_param_LR = LR_gridsearch.best_model_pridict(X_test_log)

model_KNN = KNeighborsClassifier()
neighbors = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
param_KNN = {'n_neighbors':neighbors}

KNN_gridsearch = GridSearch(X_train_log,y_train,model_KNN,param_KNN)
pred_KNN,best_param_KNN = KNN_gridsearch.best_model_pridict(X_test_log)

'''model_GBC = GradientBoostingClassifier()
r=np.random.uniform(-3,1,30)
param_GBC = {'learning_rate':10**r,
             'n_estimators':[100,200,300],
             'max_depth': [3,4,5]}
GBC_gridsearch = GridSearch(X_train_log,y_train,model_GBC,param_GBC)
pred_GBC,best_param_GBC = GBC_gridsearch.best_model_pridict(X_test_log)
'''
model_RF=RandomForestClassifier()
param_RF = {'n_estimators':[100,200,250],
            'max_depth': [6,7,8,9]}
RF_gridsearch = GridSearch(X_train_log,y_train,model_RF,param_RF)
pred_RF,best_param_RF = RF_gridsearch.best_model_pridict(X_test_log)


model_MLP = MLPClassifier()
param_MLP = {'activation' :['identity', 'logistic', 'tanh', 'relu'],
        'solver':['lbfgs', 'sgd', 'adam'],
        'learning_rate':['constant', 'invscaling', 'adaptive']}
MLP_gridsearch = GridSearch(X_train_log,y_train,model_MLP,param_MLP)
pred_MLP,best_param_MLP = MLP_gridsearch.best_model_pridict(X_test_log)
##Creating GradientBoosting model with best parameters
#final_model = GradientBoostingClassifier(**best_param_GBC)
#final_predictor = final_model.fit(X_train_log,y_train)
#print('Accuracy on train: ',final_predictor.score(X_train_log,y_train))
#print('Accuracy on test:' , final_predictor.score(X_test_log,y_test))
'''
from sklearn import metrics
#Applying log transform to whole training data
X_log = feature_transform('log',X)
# Calculating whole training data performance and AUC  
final_predictor = final_model.fit(X_log,Y)
pred = final_predictor.predict_proba(X_log)
print('Accuracy on whole training data: ',final_predictor.score(X_log,Y))
fpr, tpr, _ = metrics.roc_curve(Y, pred[:,1:2], pos_label=1)
auc = metrics.auc(fpr,tpr)
print('AUC: ',auc)
# Loading given test data
test_filename = 'test.csv'
X_Test = pd.read_csv(test_filename)

#Adding hand crafted features to given test data
X_Test['A_is_popular'] =  [1 if x >= 1000000 else 0 for x in X_Test['A_follower_count']]
X_Test['B_is_popular'] =  [1 if x >= 1000000 else 0 for x in X_Test['B_follower_count']]

#Adding hand crafted features to given test data
X_Test['A_popularity_score'] = X_Test['A_following_count'].divide(X_Test['A_follower_count'])
X_Test['B_popularity_score'] = X_Test['B_following_count'].divide(X_Test['B_follower_count'])

#Removing selected features from given test data
X_Test=X_Test.drop(cols,axis=1)

#Checking if train and test data features are in same order
X.columns == X_Test.columns

#Applying log transform to given test data
X_Test_log = feature_transform('log',X_Test)

pred_test = final_predictor.predict_proba(X_Test_log)
pred_test=pred_test[:,1:2]
pred_test
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
NFeats=10
### Load Data
DataTrain = pd.read_csv('BD/oportunidad.csv')
DataTest = pd.read_excel('BD/NewOpportunitiesList.xlsx')
CountryList = pd.read_excel('BD/Countries.xlsx')
CountryList = CountryList.set_index('ID')
###########

### Discretizate categories Supplies

Supplies_Sub= DataTrain['Supplies Subgroup'].unique()#To determine classes
Supplies_Sub.sort() # organize ranges
Supplies_Group= DataTrain['Supplies Group'].unique()#To determine classes
Supplies_Group.sort() # organize ranges
Result = DataTrain['Opportunity Result'].unique()#To determine classes
Result.sort() # organize ranges
Competitor_Label = DataTrain['Competitor Type'].unique()#To determine classes
Competitor_Label.sort() # organize ranges
for n in range(len(Supplies_Sub)):
    DataTrain['Supplies Subgroup']=DataTrain['Supplies Subgroup'].replace(Supplies_Sub[n], n) #to remplace the str class with a numerical class
for n in range(len(Supplies_Group)):
    DataTrain['Supplies Group']=DataTrain['Supplies Group'].replace(Supplies_Group[n], n) #to remplace the str class with a numerical class
for n in range(len(Result)):
    DataTrain['Opportunity Result']=DataTrain['Opportunity Result'].replace(Result[n], n) #to remplace the str class with a numerical class
for n in range(len(Competitor_Label)):
    DataTrain['Competitor Type']=DataTrain['Competitor Type'].replace(Competitor_Label[n], n) #to remplace the str class with a numerical class

### Discretizate categories Supplies (DATA TEST)

Supplies_Sub= DataTest['Supplies Subgroup'].unique()#To determine classes
Supplies_Sub.sort() # organize ranges
Supplies_Group= DataTest['Supplies Group'].unique()#To determine classes
Supplies_Group.sort() # organize ranges
Competitor_Label = DataTest['Competitor Type'].unique()#To determine classes
Competitor_Label.sort() # organize ranges
for n in range(len(Supplies_Sub)):
    DataTest['Supplies Subgroup']=DataTest['Supplies Subgroup'].replace(Supplies_Sub[n], n) #to remplace the str class with a numerical class
for n in range(len(Supplies_Group)):
    DataTest['Supplies Group']=DataTest['Supplies Group'].replace(Supplies_Group[n], n) #to remplace the str class with a numerical class
for n in range(len(Competitor_Label)):
    DataTest['Competitor Type']=DataTest['Competitor Type'].replace(Competitor_Label[n], n) #to remplace the str class with a numerical class

##To ordenate Columns, put 'Opportunity Result' First

cols = DataTrain.columns.tolist()
cols = cols[3:]+cols[:3]
DataTrain = DataTrain[cols]
DataTrain2 = DataTrain.drop('Opportunity Number',axis=1)
ResultsAllData_df = pd.DataFrame(index=['ROC','Fscore','Precision','Recall'], columns= ['All Features','Relevant Features','Relevant Features No country code'])
### BASE Line 1: Using all data (without opportunity number) a model is trained

## An adaboost classifier is trained and valitadated using a 5-fold cross validation

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

## Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
classifier = AdaBoostClassifier(n_estimators=200, random_state=0)
AllData=np.array(DataTrain2)
X_all = AllData[:,1:]
y_all = AllData[:,0]
from DrawROC import computeROC_draw
mean_auc , std_auc, All_y_predict,All_y_test = computeROC_draw(classifier,cv,X_all,y_all,'All Data')
Fscore = f1_score(All_y_test ,All_y_predict, average='binary')
recall = recall_score(All_y_test ,All_y_predict)
precision = precision_score(All_y_test ,All_y_predict)

ResultsAllData_df.loc['ROC','All Features'] = mean_auc
ResultsAllData_df.loc['Fscore','All Features'] = Fscore
ResultsAllData_df.loc['Precision','All Features'] = precision
ResultsAllData_df.loc['Recall','All Features'] = recall

###Baseline 2: Feature Selection Proccess using  Maximum Relevance — Minimum Redundancy

## use a k=5 iterations in the data to determine stable relevant features

from mrmr import mrmr_classif
SelecFeat =[] # list with the relevant features in each fold
for train_index, test_index in cv.split(X_all,y_all):
    X_eval = DataTrain2.iloc[train_index,:] # select test data to find relevant features
    FeaturesFold = mrmr_classif(X=X_eval.iloc[:,1:], y=X_eval.iloc[:,0], K=NFeats)
    SelecFeat.append(FeaturesFold)

## To find most Commun features in each fold
# (if a relevant feature is selected in at least four folds then is relevant for the data)
TopFeatures = np.unique(np.array(SelecFeat))
import itertools as it
listFeats = list(it.chain(*SelecFeat))
CountTop = []
for top in TopFeatures:
    CountF =listFeats.count(top)
    CountTop.append(CountF)
SelecFeat = TopFeatures[np.array(CountTop)>3] # if the feature is common in at least four folds is selected
print('Best Features :'+str(SelecFeat))

### Evaluate the performance by exploring of the selected features using the country index in the train data

for test in range(2,len(SelecFeat),2):
    Selection = SelecFeat[:test]
    X = np.array(DataTrain2.loc[:,Selection])
    y = np.array(DataTrain2.loc[:,'Opportunity Result'])
    mean_auc, std_auc,All_y_predict,All_y_test  = computeROC_draw(classifier, cv, X, y, str(test)+'Features, all Countries')
    Fscore = f1_score(All_y_test, All_y_predict, average='binary')
    recall = recall_score(All_y_test, All_y_predict)
    precision = precision_score(All_y_test, All_y_predict)

ResultsAllData_df.loc['ROC','Relevant Features']=mean_auc
ResultsAllData_df.loc['Fscore','Relevant Features']=Fscore
ResultsAllData_df.loc['Precision','Relevant Features'] = precision
ResultsAllData_df.loc['Recall','Relevant Features'] = recall

### Baseline 3: Train a model by selecting relevant features in the data without use country index (Naive decision)

SelecFeat =[]
for train_index, test_index in cv.split(X_all,y_all):
    X_eval = DataTrain2.iloc[train_index,:]
    X_eval = X_eval.drop('Country_Code',axis=1)
    FeaturesFold = mrmr_classif(X=X_eval.iloc[:,1:], y=X_eval.iloc[:,0], K=NFeats)
    SelecFeat.append(FeaturesFold)

##To find most Commun features in each fold

TopFeatures = np.unique(np.array(SelecFeat))
import itertools as it
listFeats = list(it.chain(*SelecFeat))
CountTop = []
for top in TopFeatures:
    CountF =listFeats.count(top)
    CountTop.append(CountF)
SelecFeat = TopFeatures[np.array(CountTop)>3] # if the feature is common in at least four folds is selected
print('Best Features:'+str(SelecFeat))

## Evaluate the performance by exploring of the selected features without the country vector

for test in range(2,len(SelecFeat),2):
    Selection = SelecFeat[:test]
    X = np.array(DataTrain2.loc[:,Selection])
    y = np.array(DataTrain2.loc[:,'Opportunity Result'])
    mean_auc, std_auc , All_y_predict,All_y_test= computeROC_draw(classifier, cv, X, y, str(test)+'Features, all Countries')
    Fscore = f1_score(All_y_test, All_y_predict, average='binary')
    recall = recall_score(All_y_test, All_y_predict)
    precision = precision_score(All_y_test, All_y_predict)
ResultsAllData_df.loc['ROC','Relevant Features No country code']=mean_auc
ResultsAllData_df.loc['Fscore','Relevant Features No country code']=Fscore
ResultsAllData_df.loc['Precision','Relevant Features No country code'] = precision
ResultsAllData_df.loc['Recall','Relevant Features No country code'] = recall

### Baseline Countries: Using relevant features train an independent model per country

Country_codes = DataTrain2['Country_Code'].unique()
DataTrain_Country = DataTrain2.copy()
DataTrain_Country = DataTrain_Country.set_index('Country_Code')
Matrix_dataframe = pd.DataFrame(columns=CountryList.loc[:,'Country'],index=CountryList.loc[:,'Country'])
Matrix_dataframe_Fs = pd.DataFrame(columns=CountryList.loc[:,'Country'],index=CountryList.loc[:,'Country'])
Matrix_dataframe_Pr = pd.DataFrame(columns=CountryList.loc[:,'Country'],index=CountryList.loc[:,'Country'])
Matrix_dataframe_Re = pd.DataFrame(columns=CountryList.loc[:,'Country'],index=CountryList.loc[:,'Country'])

Results_df = pd.DataFrame(index=CountryList.loc[:,'Country'], columns= ['Baseline Countries','DataIntegration'])
Results_df_Fs = pd.DataFrame(index=CountryList.loc[:,'Country'], columns= ['Baseline Countries','DataIntegration'])
Results_df_Pr = pd.DataFrame(index=CountryList.loc[:,'Country'], columns= ['Baseline Countries','DataIntegration'])
Results_df_Re = pd.DataFrame(index=CountryList.loc[:,'Country'], columns= ['Baseline Countries','DataIntegration'])


for code in Country_codes:
    DataTrain_Country2 = DataTrain_Country.loc[code]
    Selection = SelecFeat
    X = np.array(DataTrain_Country2.loc[:, Selection])
    y = np.array(DataTrain_Country2.loc[:, 'Opportunity Result'])
    mean_auc, std_auc,All_y_predict,All_y_test = computeROC_draw(classifier, cv, X, y, 'Best Features '+CountryList.loc[code,'Country'])
    Fscore = f1_score(All_y_test, All_y_predict, average='binary')
    precision = precision_score(All_y_test, All_y_predict)
    recall = recall_score(All_y_test, All_y_predict)
    Matrix_dataframe.loc[CountryList.loc[code, 'Country'], CountryList.loc[code, 'Country']]= mean_auc
    Matrix_dataframe_Fs.loc[CountryList.loc[code, 'Country'], CountryList.loc[code, 'Country']]= Fscore
    Matrix_dataframe_Re.loc[CountryList.loc[code, 'Country'], CountryList.loc[code, 'Country']]= recall
    Matrix_dataframe_Pr.loc[CountryList.loc[code, 'Country'], CountryList.loc[code, 'Country']]= precision

    Results_df.loc[CountryList.loc[code, 'Country'], 'Baseline Countries'] = mean_auc
    Results_df_Fs.loc[CountryList.loc[code, 'Country'], 'Baseline Countries'] = Fscore
    Results_df_Pr.loc[CountryList.loc[code, 'Country'], 'Baseline Countries'] = precision
    Results_df_Re.loc[CountryList.loc[code, 'Country'], 'Baseline Countries'] = recall

### Blind Evaluation the test Data using the relevant features in a model trained with data from all countries
classifier_Final = AdaBoostClassifier(n_estimators=200, random_state=0)
from sklearn import preprocessing
X = np.array(DataTrain2.loc[:, SelecFeat])
y = np.array(DataTrain2.loc[:, 'Opportunity Result'])
scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X)
classifier_Final.fit(X_train, y)
# Prepare test data
X_test = DataTest.loc[:,SelecFeat]
X_test = scaler.transform(X_test)
Y_preds = classifier_Final.predict(X_test)

print('Won Predictions:'+str(np.sum(Y_preds==1)))
print('Lose Predictions:'+str(np.sum(Y_preds==0)))
print('End')


### Baseline DataIntegration: Loking for countries with similar decision proccess
from DrawROC import  computeROC_draw_single
Country_codes = DataTrain2['Country_Code'].unique()
DataTrain_Country = DataTrain2.copy()
DataTrain_Country = DataTrain_Country.set_index('Country_Code')
for code in Country_codes:
    DataTrain_Country2 = DataTrain_Country.loc[code]
    Selection = SelecFeat
    X_train = np.array(DataTrain_Country2.loc[:, Selection])
    y_train = np.array(DataTrain_Country2.loc[:, 'Opportunity Result'])
    DataTest_Country = DataTrain_Country.drop(index=code) # to remove the train country
    Country_codes_pop = np.unique(DataTest_Country.index)
    for code2 in Country_codes_pop:
        DataTest_Country3 = DataTrain_Country.loc[code2]
        X_test = np.array(DataTest_Country3.loc[:, Selection])
        y_test = np.array(DataTest_Country3.loc[:, 'Opportunity Result'])
        mean_auc, std_auc, All_y_predict,All_y_test = computeROC_draw_single(classifier,X_train,y_train,X_test,y_test, 'Train '+CountryList.loc[code,'Country']+' Test in'+CountryList.loc[code2,'Country'])
        Fscore = f1_score(All_y_test, All_y_predict, average='binary')
        precision = precision_score(All_y_test, All_y_predict)
        recall = recall_score(All_y_test, All_y_predict)
        Matrix_dataframe_Pr.loc[CountryList.loc[code, 'Country'], CountryList.loc[code2, 'Country']] = precision

        Matrix_dataframe_Re.loc[CountryList.loc[code, 'Country'], CountryList.loc[code2, 'Country']] = recall

        Matrix_dataframe_Fs.loc[CountryList.loc[code, 'Country'], CountryList.loc[code2, 'Country']] = Fscore
        Matrix_dataframe.loc[CountryList.loc[code, 'Country'], CountryList.loc[code2, 'Country']] = mean_auc

Matrix_dataframe = Matrix_dataframe_Re.copy()

## To define a treshold for the obtained classification

Country_relationships = Matrix_dataframe.copy()
for country in range(len(Matrix_dataframe)):
    Country_relationships.iloc[:,country] = Matrix_dataframe.iloc[country,country]-Matrix_dataframe.iloc[:,country]<0.0005


### Results for countries relationships

sns.heatmap(Matrix_dataframe.astype(np.float64), annot=True,fmt=".2f")
plt.show()
sns.heatmap(Country_relationships.astype(np.float64), annot=True,fmt=".2f")
plt.show()

### To Create models using countries relations

CountryList_codes = CountryList.copy()
from DrawROC import computeROC_draw_integration
for country_train in Country_relationships.columns:
    array_relations = Country_relationships.loc[:,country_train].values
    TrainCountries = CountryList[array_relations]
    Data_Env = DataTrain_Country.loc[TrainCountries.index.values]

    index_country = np.argmax(CountryList == country_train) + 1
    try:
        Data_To_Fold = Data_Env.loc[index_country]
        Data_Env = Data_Env.drop(index=index_country)
        X_int = np.array(Data_Env.loc[:, Selection])
        y_int = np.array(Data_Env.loc[:, 'Opportunity Result'])
        X = np.array(Data_To_Fold.loc[:, Selection])
        y = np.array(Data_To_Fold.loc[:, 'Opportunity Result'])
        mean_auc, std_auc, All_y_predict, All_y_test= computeROC_draw_integration(classifier, cv, X_int, y_int, X, y, 'data integration for:'+country_train)
        Fscore = f1_score(All_y_test, All_y_predict, average='binary')
        precision = precision_score(All_y_test, All_y_predict)
        recall = recall_score(All_y_test, All_y_predict)
        Results_df.loc[country_train, 'DataIntegration'] = mean_auc
        Results_df_Fs.loc[country_train, 'DataIntegration'] = Fscore
        Results_df_Pr.loc[country_train, 'DataIntegration'] = precision
        Results_df_Re.loc[country_train, 'DataIntegration'] = recall
    except:
        print('Data No exist')

### Show all Results

sns.heatmap(ResultsAllData_df.astype(np.float64), annot=True,fmt=".2f")
plt.show()
print('Results AUC')
sns.heatmap(Results_df.astype(np.float64), annot=True,fmt=".2f")
plt.show()
print('Results Fscore')
sns.heatmap(Results_df_Fs.astype(np.float64), annot=True,fmt=".2f")
plt.show()
ax = plt.axes()
sns.heatmap(Results_df_Fs, ax = ax)
ax.set_title('Fscore')
plt.show()

### Blind Evaluation the test Data using the relevant features and the data integration

DataTest = DataTest.set_index('Country')
All_yPreds=np.array([])
for country_train in Country_relationships.columns:
    try:
        array_relations = Country_relationships.loc[:,country_train].values
        TrainCountries = CountryList[array_relations]
        Data_Train_Blind = DataTrain_Country.loc[TrainCountries.index.values]

        X_int = np.array(Data_Train_Blind.loc[:, Selection])
        y_int = np.array(Data_Train_Blind.loc[:, 'Opportunity Result'])
        classifier_Final = AdaBoostClassifier(n_estimators=200, random_state=0)
        scaler = preprocessing.StandardScaler().fit(X_int)
        X_train = scaler.transform(X_int)
        classifier_Final.fit(X_train, y_int)
        X_test = DataTest.loc[:, SelecFeat]
        X_test = X_test.loc[country_train]
        X_test = scaler.transform(X_test)
        Y_preds = classifier_Final.predict(X_test)
        All_yPreds = np.concatenate((All_yPreds,Y_preds))
    except:
        print('No data')
print('Won Predictions:'+str(np.sum(All_yPreds==1)))
print('Lose Predictions:'+str(np.sum(All_yPreds==0)))
print('End')
# print('Same Procces discriminating By Country')
#
# for code in Country_codes:
#     DataTrain_Country2 = DataTrain_Country.loc[code]
#     Selection = SelecFeat
#     X = np.array(DataTrain_Country2.loc[:, Selection])
#     y = np.array(DataTrain_Country2.loc[:, 'Opportunity Result'])
#     mean_auc, std_auc = computeROC_draw(classifier, cv, X, y, 'Best Features '+CountryList.loc[code,'Country'])
#
#     ## BASE Line 1: Using all data (less opportunity number) without differenciate countries
#
#
#     # Run classifier with cross-validation and plot ROC curves
#     cv = StratifiedKFold(n_splits=5)
#     classifier = AdaBoostClassifier(n_estimators=100, random_state=0)
#     AllData = np.array(DataTrain_Country2)
#     X = AllData[:, 1:]
#     y = AllData[:, 0]
#
#     from DrawROC import computeROC_draw
#
#     mean_auc, std_auc = computeROC_draw(classifier, cv, X, y, 'All Data'+CountryList.loc[code,'Country'])
#
#     ##Feature Selection
#     # use a k iteration in the data to determine stable relevant features
#     SelecFeat = []
#     import pymrmr
#     from mrmr import mrmr_classif
#
#     for train_index, test_index in cv.split(X, y):
#         X_eval = DataTrain_Country2.iloc[train_index, :]
#        # X_eval = X_eval.drop('Country_Code', axis=1)
#         FeaturesFold = mrmr_classif(X=X_eval.iloc[:, 1:], y=X_eval.iloc[:, 0], K=NFeats)
#         #        pymrmr.mRMR(X_eval, 'MIQ',14)
#         SelecFeat.append(FeaturesFold)
#
#     ##Find most Commun features
#
#     TopFeatures = np.unique(np.array(SelecFeat))
#     import itertools as it
#
#     listFeats = list(it.chain(*SelecFeat))
#     CountTop = []
#     for top in TopFeatures:
#         CountF = listFeats.count(top)
#         CountTop.append(CountF)
#
#     ## Evaluate the performance by explorating of the selected feaures
#     SelecFeat = TopFeatures[np.array(CountTop) > 3]
#     print('Best Features:' + str(SelecFeat))
#     Selection = SelecFeat
#     #    print(Selection)
#     X = np.array(DataTrain_Country2.loc[:, Selection])
#     y = np.array(DataTrain_Country2.loc[:, 'Opportunity Result'])
#     mean_auc, std_auc = computeROC_draw(classifier, cv, X, y,  'Features Selecction'+CountryList.loc[code,'Country'])
#
# print('end')
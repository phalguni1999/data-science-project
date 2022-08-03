# for the data manipulation
import numpy as np
import pandas as pd

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for interactive
from ipywidgets import interact

# read the dataset
data = pd.read_csv('C:/Users/HP\OneDrive/Desktop/cropharvestdata.csv')

# shape of the dataset
print("shape of the data set is :", data.shape)

# check the first five row of the dataset
data1 = data.head()

# check if there is any missing value predent in dataset
data.isnull()

data.isnull().sum()


#fill the null values with the mean value of that column where the null value is present . 
data.fillna(data['K'].mean())

# check the crops present in the dataset
data['label'].value_counts()

# let's check the summary of all crops present in datset
print("average ratio of Nitrogen in soil : {0:.2f}" .format(data['N'].mean()))
print("average ratio of phosphorous in soil : {0:.2f}".format(data['P'].mean()))
print("average ratio of potassium in soil : {0:.2f}".format(data['K'].mean()))
print("average temperature present5 in soil in celcius : {0:.2f}".format(data['temperature'].mean()))
print("average relative humidity in % : {0:.2f}".format(data['humidity'].mean()))
print("average PH value present in soil : {0:.2f}".format(data['ph'].mean()))
print("average rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))


#let's check the summary statisstics for each of the crop present in the dataset


def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    print("--------------------------------------------------")
    print("statistics for nitrogen")
    print("minimum nitrogen required :" , x['N'].min())
    print("average nitrogedn required :", x['N'].mean())
    print("maximum nitrogen required :" , x['N'].max())
    
    print("--------------------------------------------------")
    print("statistics fot phosphorous")
    print("minimum phosphorous required :" , x['P'].min())
    print("average phosphorous required :" , x['P'].mean())
    print("maximum phosphorous required :" , x['P'].max())
    
    print("--------------------------------------------------")
    print("statistics for potassium")
    print("minimum potassium required :" , x['K'].min())
    print("average potassium required :" , x['K'].mean())
    print("maximum potassium required :" , x['K'].max())
    
    print("--------------------------------------------------")
    print("statistics for temperature")
    print("minimum temprature required : {0:.2f}".format(x['temprature'].min()))
    print("average temprature required : {0:.2f}".format(x['temprature'].mean()))
    print("maximum temprature required :{0:.2f}".format(x['temprature'].max()))
    
    print("--------------------------------------------------")
    print("statistics for humidity")
    print("minimum humidity required :{0:.2f}".format(x['humidity'].min()))
    print("average humidity required :{0:.2f}".format(x['humidity'].mean()))
    print("maximum humidity required :{0:.2f}".format(x['humidity'].max()))
    
    print("--------------------------------------------------")
    print("statistics for ph")
    print("minimum ph required :{0:.2f}".format(x['ph'].min()))
    print("average ph required :{0:.2f}".format(x['ph'].mean()))
    print("maximum ph required :{0:.2f}".format(x['ph'].max()))
    
    print("--------------------------------------------------")
    print("statistics for rainfall")
    print("minimum rainfall required :{0:.2f}".format(x['rainfall'].min()))
    print("average rainfall required :{0:.2f}".format(x['rainfall'].mean()))
    print("maximum rainfall required :{0:.2f}".format(x['rainfall'].max()))
    
    
# let's compare the average requirement  for each crops with average conditions


@interact
def compare(condition = ['N' , 'P' , 'K' , 'temperature' , 'ph' , 'humidity' , 'rainfall']):
    print("average vale fot" , condition , "is {0:.2f}".format(data[condition].mean()))
    print("----------------------------------------------------")
    print("rice : {0:.2f}".format(data[(data['label'] == 'rice')][condition].mean()))
    print("oilseed : {0:.2f}".format(data[(data['label'] == 'oilseed')][condition].mean()))
    print("pulses : {0:.2f}".format(data[(data['label'] == 'pulses')][condition].mean()))    
    print("turmeric : {0:.2f}".format(data[(data['label'] == 'turmeric')][condition].mean()))
    print("potato : {0:.2f}".format(data[(data['label'] == 'potato')][condition].mean()))
    print("moathbeans : {0:.2f}".format(data[(data['label'] == 'moathbeans')][condition].mean()))
    print("sugarcane : {0:.2f}".format(data[(data['label'] == 'sugarcane')][condition].mean()))
    print("jute : {0:.2f}".format(data[(data['label'] == 'jute')][condition].mean()))
    print("moatbeans : {0:.2f}".format(data[(data['label'] == 'moatbeans')][condition].mean()))
    
    
#let's make this function more intuitive
#check all the crops which require greater than and less than each condition


@interact
def compare(conditions = ['N' , 'P' , 'K' , 'temperature' , 'ph' , 'humidity' , 'rainfall']):
    print("crops which require greater than average" , conditions ,'\n')
    print(data[data[conditions] > data[conditions].mean()]['label'].unique())
    print("-----------------------------------------------------")
    print("crops which requires less than the average" ,conditions , '\n')
    print(data[data[conditions] <= data[conditions].mean()]['label'].unique())
    
    
#plotting the distribution chart

plt.subplot(2,4,1)
sns.distplot(data['N'] , color = 'darkblue')
plt.xlabel('ratio of nitrogen' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()

plt.subplot(2,4,2)
sns.distplot(data['P'] , color = 'yellow')
plt.xlabel('ratio of phosphorous' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()

plt.subplot(2,4,4)
sns.distplot(data['K'] , color = 'green')
plt.xlabel('ratio of potassium' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()

plt.subplot(2,4,6)
sns.distplot(data['temperature'] , color = 'black')
plt.xlabel('temperature' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()    

plt.subplot(2,4,8)
sns.distplot(data['ph'] , color = 'grey')
plt.xlabel('ph' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()

plt.subplot(2,4,3)
sns.distplot(data['humidity'] , color = 'orange')
plt.xlabel('humidity' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()

plt.subplot(2,4,7)
sns.distplot(data['rainfall'] , color = 'red')
plt.xlabel('rainfall' , fontsize = 12)
plt.grid()

plt.suptitle('distribution for agricultural conditions' , fontsize = 20)
plt.show()



# let's findout some interesting facts

print("some interesting patterens")
print("----------------------------------------------")
print("crops which requires very high ratio of nitrogen content in soil:" , data[data['N']>80]['label'].unique())
print("crops which requires very high ratio of phosphorous content in soil:" , data[data['P']>90]['label'].unique())
print("crops which requires very high ratio of nitrogen content in soil:" , data[data['N']>95]['label'].unique())
print("crops which requires high rainfall:" , data[data['rainfall'] > 120]['label'].unique())
print("crops which requires very low temperature:" , data[data['temperature'] < 5]['label'].unique())
print("crops which requires very high temperature:" , data[data['temperature'] > 10]['label'].unique())
print("crops which requires very low humidity:" , data[data['humidity'] < 5]['label'].unique())
print("crops which requires very low ph:" , data[data['ph'] < 4]['label'].unique())
print("crops which requires very high ph:" , data[data['ph'] > 8]['label'].unique())



#let's understand which crops can only grown in summerseason , winterseason , rainyseason

print("summer crops")
print(data[(data['temperature']>20) & (data['humidity']>30)]['label'].unique())
print("------------------------------------")
print("winter crops")
print(data[(data['temperature']>10) & (data['humidity']>20)]['label'].unique())
print("------------------------------------")
print("rainy crops")
print(data[(data['rainfall']>100) & (data['humidity']>30)]['label'].unique())

# EDA process is completed .


#M.L part

from sklearn.cluster import KMeans

#removing the label data from the dataset
data2 = data.drop(['label'],axis = 1)

print(data2)

print(data.shape)
print(data2.shape)

X = data.iloc[:,[3,4,5,6]]

print(X.shape)

#plotting the elbow chart
plt.rcParams['figure.figsize'] = (10,4)
wcss = []

for i in range(1,8):
    km = KMeans(n_clusters=i , init='k-means++' , max_iter=300 , n_init=10 , random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
    
    
#let's plot the result
plt.plot(range(1,8),wcss)
plt.title('THE ELBOW CHART' , fontsize = 18)
plt.xlabel('no. of clusters' , fontsize = 15)
plt.ylabel('wcss' , fontsize = 15)
plt.show()


#let's implement the KMeans algorithm to performing clusdtering analysis
km = KMeans(n_clusters=3 , init='k-means++' , max_iter=300 , n_init=10 , random_state = 0)
y_means = km.fit_predict(X)

#let's findout the result
a = data['label']
y_means = pd.DataFrame(y_means)
print(y_means)
z = pd.concat([y_means , a] , axis = 1)
z = z.rename(columns = {0: 'cluster'})


#let's check the clusters for each crop

print("let's check the result after applying the kmeans clustering analysis \n")
print("crops in first cluster:"  , z[z['cluster'] == 0]['label'].unique())
print("------------------------------------------------------------------")
print("crops in second cluster:"  , z[z['cluster'] == 1]['label'].unique())
print("------------------------------------------------------------------") 
print("crops in third cluster:"  , z[z['cluster'] == 2]['label'].unique())


#let's split the data in dataset for the predicting modelling

y = data['label']
x = data.iloc[:,[3,4,5,6]]

print("shape of x:" , x.shape)
print("shape of y:" , y.shape)


#let's create training and testing set for the validation result

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , random_state = 0)

print("shape of x_train :" , x_train.shape)
print("shape of x_test :", x_test.shape)
print("shape of y_train : ", y_train.shape)
print("shape of y_test : ", y_test.shape)


# let's create a predictive model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)



#let's evaluate the model performance
from sklearn.metrics import confusion_matrix


#let's print the confusionmatrix first
plt.rcParams['figure.figsize'] = (10,10)
cm = confusion_matrix(y_test , y_pred)
sns.heatmap(cm , annot = True , cmap = 'Wistia')
plt.title('confusion matrix for logistic regrassion' , fontsize = 25)
plt.show()


#let's print the3 classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test , y_pred)
print(cr)


data.head()


#let's check the prediction is correct or not

prediction = model.predict([[  20 , 81 , 7 , 201]])

print("the suggest crop for given climatic condition is:" , prediction)


#end......
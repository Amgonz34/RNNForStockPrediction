#import needed packages
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")
import numpy as np

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers


#read CSV's into data frames
df = pd.read_csv("EXStockDataWithNewsData.csv")
df1 = pd.read_csv("OgStockDataWithNewsData.csv")
df2 = pd.read_csv("ExtrackedStockData.csv")
df3 = pd.read_csv("StockData.csv")



# funtion to clean data as needed
def clean(df, df1, df2, df3):
    df = df.drop('Unnamed: 0', axis=1)
    df1 = df1.drop('Unnamed: 0', axis=1)

    org_dates = df['Date']

    df['Date'] = pd.to_datetime(df["Date"])
    df1['Date'] = pd.to_datetime(df1["Date"])
    df2['Date'] = pd.to_datetime(df2["Date"])
    df3['Date'] = pd.to_datetime(df2["Date"])

    df3 = df3.drop([2476])
    df = df.set_index('Date')
    df1 = df1.set_index('Date')
    df2 = df2.set_index('Date')
    df3 = df3.set_index('Date')

    df = df.drop('nextDayOpen', axis=1)
    df1 = df1.drop('nextDayOpen', axis=1)
    df2 = df2.drop('nextDayOpen', axis=1)
    df3 = df3.drop('nextDayOpen', axis=1)

    df1 = df1.drop('Time', axis=1)
    
    return df, df1, df2, df3

df, df1, df2, df3 = clean(df, df1, df2, df3) 

#seting times size of LSTM and ration of train test. 
TimeSize = 50
ratio = .9 

#function used to reformat the data so it is usable with LSTM 
def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

#function that builds RNN
def buildModel(X_train, y_train): 
    model = keras.Sequential()
    
    #input layer
    model.add(LSTM(units = 500, return_sequences = True, input_shape = (X_train.shape[1], 1)))    
    model.add(Dropout(0.1))    
    
    #Hidden layer
    model.add(LSTM(units = 250))    
    model.add(Dropout(0.1))    
    model.add(Dense(units = 50)) 
    # Output layer
    model.add(Dense(units = 1))
    model.add(Activation('linear', name='linear_output'))

    # Compiling the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the model to the Training set
    history = model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split=.30)
        
    return history, model
   
#Funtion to plot results
def plot(org_y, predicted_y):
    # plot the results 
    plt.figure(figsize = (12,6))
    plt.plot(org_y, color = 'red', label = 'Real Amazon Stock Price')
    plt.plot(predicted_y, color = 'blue', label = 'Predicted Amazon Stock Price')
    plt.title('Amazon Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()  
    
#-----------------------------------------------------------------------------

'''
The original plan was to build a function to be used for the normalization and reshaping 
of the train and test datasets for each dataset but I kept running into errors when trying 
to use fit.transform to unnormalize the results from the model. Due to this the code that 
sandwiched the model training for each data set for this was just rewritten to make development quicker. 
'''

#------------------------------------------------------------------------------

#normilize and shape first train data set
data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(df)
dfn = pd.DataFrame(data_normalised, columns=df.columns, index=df.index)
    
Y_normaliser = preprocessing.MinMaxScaler()
Y_normaliser.fit_transform(df[["Open"]])
    
    
size = df.shape[0]* ratio
    

dfTrain = dfn[:int(size)]
dfTest = dfn[(int(size)):]
    
X_train, y_train = create_dataset(dfTrain.to_numpy(), TimeSize)    
X_train = np.reshape(X_train, (X_train.shape[0], TimeSize, 1))    

#train first model
history, model = buildModel(X_train, y_train)    
 

 #normilize first test data set  
X_test, y_test = create_dataset(dfTest.to_numpy(), TimeSize)
y_test = y_test.reshape(-1,1)    
    
org_y = Y_normaliser.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], TimeSize, 1))

# Predict the prices with the model
predicted_y = model.predict(X_test)
predicted_y = Y_normaliser.inverse_transform(predicted_y)

org_ydf = pd.DataFrame(data=org_y, columns=["Real Opens"])

dfExSN = pd.DataFrame(data=predicted_y, columns=["Predicted Opens Ex News, Stock"])

#----------------------------------------------------------------------------------------

#normilize and shape secound train data set
data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(df)
dfn = pd.DataFrame(data_normalised, columns=df.columns, index=df.index)
    
Y_normaliser = preprocessing.MinMaxScaler()
Y_normaliser.fit_transform(df[["Open"]])
    
    
size = df.shape[0]* ratio
    

dfTrain = dfn[:int(size)]
dfTest = dfn[(int(size)):]
    
X_train, y_train = create_dataset(dfTrain.to_numpy(), TimeSize)    
X_train = np.reshape(X_train, (X_train.shape[0], TimeSize, 1))

#train secound model
history1, model1 = buildModel(X_train, y_train)

#normilize and shape secound test data set
X_test, y_test = create_dataset(dfTest.to_numpy(), TimeSize)
y_test = y_test.reshape(-1,1)    
    
org_y = Y_normaliser.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], TimeSize, 1))

# Predict the prices with the model
predicted_y = model1.predict(X_test)
predicted_y = Y_normaliser.inverse_transform(predicted_y)

org_ydf1 = pd.DataFrame(data=org_y, columns=["Real Opens"])

dfExN = pd.DataFrame(data=predicted_y, columns=["Predicted Opens Ex News"]) 
    

#-------------------------------------------------------------------------------

#normilize and shape third train data set
data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(df)
dfn = pd.DataFrame(data_normalised, columns=df.columns, index=df.index)
    
Y_normaliser = preprocessing.MinMaxScaler()
Y_normaliser.fit_transform(df[["Open"]])
    
    
size = df.shape[0]* ratio
    

dfTrain = dfn[:int(size)]
dfTest = dfn[(int(size)):]
    
X_train, y_train = create_dataset(dfTrain.to_numpy(), TimeSize)    
X_train = np.reshape(X_train, (X_train.shape[0], TimeSize, 1))

#train third model
history2, model2 = buildModel(X_train, y_train)

#normilize and shape third test data set

X_test, y_test = create_dataset(dfTest.to_numpy(), TimeSize)
y_test = y_test.reshape(-1,1)    
    
org_y = Y_normaliser.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], TimeSize, 1))

# Predict the prices with the model
predicted_y = model2.predict(X_test)
predicted_y = Y_normaliser.inverse_transform(predicted_y)

org_ydf2 = pd.DataFrame(data=org_y, columns=["Real Opens"])

dfExS = pd.DataFrame(data=predicted_y, columns=["Predicted Opens EX Stock"])

#--------------------------------------------------------------------------------------

#normilize and shape fourth train data set

data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(df)
dfn = pd.DataFrame(data_normalised, columns=df.columns, index=df.index)
    
Y_normaliser = preprocessing.MinMaxScaler()
Y_normaliser.fit_transform(df[["Open"]])
    
    
size = df.shape[0]* ratio
    

dfTrain = dfn[:int(size)]
dfTest = dfn[(int(size)):]
    
X_train, y_train = create_dataset(dfTrain.to_numpy(), TimeSize)    
X_train = np.reshape(X_train, (X_train.shape[0], TimeSize, 1))


#train fourth model
history3, model3 = buildModel(X_train, y_train)


#normilize and shape fourth test data set
X_test, y_test = create_dataset(dfTest.to_numpy(), TimeSize)
y_test = y_test.reshape(-1,1)    
    
org_y = Y_normaliser.inverse_transform(y_test)

# reshape it [samples, time steps, features]
X_test = np.reshape(X_test, (X_test.shape[0], TimeSize, 1))

# Predict the prices with the model
predicted_y = model3.predict(X_test)
predicted_y = Y_normaliser.inverse_transform(predicted_y)

org_ydf3 = pd.DataFrame(data=org_y, columns=["Real Opens"])

dfNEx = pd.DataFrame(data=predicted_y, columns=["Predicted Opens Stock data"])

#------------------------------------------------------------------------------------


#print model visulizations
plot(org_ydf, dfExSN)#all

plot(org_ydf1, dfExN)#news

plot(org_ydf2, dfExS) #stocks

plot(org_ydf3, dfNEx) #basic


#save results
frames = (org_ydf, dfExSN, dfExN, dfExS, dfNEx)
ResultData = pd.concat(frames, axis=1, sort=False)
ResultData.to_csv('ResultData.csv')






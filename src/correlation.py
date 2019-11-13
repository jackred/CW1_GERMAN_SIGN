### DATA MINING AND MACHINE LEARNING COURSEWORK 1
### question 6  - cropping dataset according to correlation rating

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def _attribute_to_row(dataset):
    matrix = (list(zip(*reversed(dataset))))
    return matrix


def _correlation_matrix(filename, filename2, size):
    # data = _attribute_to_row(pd.read_csv(filename)) # load the dataset to evaluate
    data = pd.read_csv(filename)
    # label = attribute_to_row(pd.read_csv(filename2))
    # load the label dataset that will be used to calculate correlation
    label = pd.read_csv(filename2)
    dataset = np.concatenate((data, label),axis=1)  # concatenate the dataset and the label
    dataset = pd.DataFrame(dataset, index=dataset[:, 0])
    #print(dataset)
     #dataset.iloc[:, 1:-1]
    #label_encoder = LabelEncoder()
    #dataset.iloc[:, 0] = label_encoder.fit_transform(dataset.iloc[:, 0]).astype('float64')
    corr = dataset.corr()  # calculate the corr of all the different pairs of row in the dataset
    best = np.zeros(size)  # initialise the array that will store the best "size" correlation
    index = np.zeros(size)  # initialise the array that will be return with the index of the best correlated attributes
    datasize = len(dataset)  # store the size of the edge of the correlation square
    for i in range(datasize):
        # last columns contains the correlation with the label
        # reset temp modified
        modified = False
        print("A",modified)
        for j in range(len(best)):
            if best[j] < corr[i][np.sqrt(corr.size)-1] and modified==False:
                best[j] = corr[i][np.sqrt(corr.size)-1]
                index[j] = i
                modified = True
    print(index)



def _correlation_matrix2(filename,filename2,size):
    # data = _attribute_to_row(pd.read_csv(filename)) # load the dataset to evaluate
    data = pd.read_csv(filename)
    # label = attribute_to_row(pd.read_csv(filename2))
    # load the label dataset that will be used to calculate correlation
    label = pd.read_csv(filename2)
    dataset = np.concatenate((data, label), axis=1)  # concatenate the dataset and the label
    dataset=pd.DataFrame(dataset,index=dataset[:,0])
    corr = dataset.corr()  # calculate the corr of all the different pairs of row in the dataset
    print(type(corr),corr[np.sqrt(corr.size)-1][1])


def _concatenate_corr_matrix(filename,array):
    indexes=np.reshape(array,1,1)
    indexes=list(set(indexes))
    with open(path+filename, 'r') as f:
        h = f.readline()








_correlation_matrix("x_train_gr_smpl.csv", "y_train_smpl_0.csv", 25)


path = '../data/'
prefix = 'y_train_smpl_'
extension = '.csv'
main_file = 'x_train_gr_smpl.csv'
files = [main_file, prefix[0:-1]+extension]
for i in range(10):
    files.append(prefix+str(i)+extension)

array = [2, 5, 10]
index10 = []
index50 = []
index100 = []
for name in files:
    index10.append(_correlation_matrix(main_file, name, 2))
    index50.append(_correlation_matrix(main_file, name, 5))
    index100.append(_correlation_matrix(main_file, name, 10))



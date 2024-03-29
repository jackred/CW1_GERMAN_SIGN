from arg import kmeans_args
from sklearn.cluster import KMeans, FeatureAgglomeration, SpectralClustering, MiniBatchKMeans, Birch
from helper import pre_processed_data, pre_processed_label, matrix_confusion, \
    create_images_from_rows
import numpy as np
from preprocess import mean_image

'''
  In this file, a few cluster algorithm are implemented :
    - KMeans
    - Feature Agglomeration
    - Spectral Clustering
    - Mini Batch Kmeans
'''

def birch(data_train, data_test, label_train, label_test, args):
    print('birch')
    birch = Birch(n_clusters=10).fit(data_train)
    predict = birch.predict(data_test)
    print('birch done')
    compare_class(predict, label_test)
    if args.create_mean:
        create_images_from_rows('bi', mean_image(predict, data_test))

def find_highest(old):
    arr = np.array([])
    for elem in old:
        result = np.where(elem == np.amax(elem))
        arr = np.append(arr, int(result[0][0]))
    return arr

def featureagglomeration(data_train, data_test, label_train, label_test, args):
    print('feature agglomeration')
    FA = FeatureAgglomeration(n_clusters=10).fit(data_train)
    transformation = FA.transform(data_test)
    agglomeration = find_highest(transformation)
    print('feature agglomeration done')
    compare_class(agglomeration, label_test)
    if args.create_mean:
        create_images_from_rows('fa', mean_image(agglomeration, data_test))

def spectralclustering(data_train, data_test, label_train, label_test, args):
    print('spectral clustering')
    SC = SpectralClustering(n_clusters=10, affinity='nearest_neighbors').fit(data_train)
    predict = SC.fit_predict(data_test)
    print('spectral clustering done')
    compare_class(predict, label_test)
    if args.create_mean:
        create_images_from_rows('SC', mean_image(SC.labels_, data_test))

def minibatchkmeans(data_train, data_test, label_train, label_test, args):
    print('mini batch kmeans')
    switch_choice = {'k': lambda: 'k-means++', 'r': lambda: 'random',
                     'm': lambda: mean_image(label_train, data_train)}
    MKMeans = MiniBatchKMeans(n_clusters=10, init=switch_choice[args.init](), n_init=10).fit(data_train)
    predict = MKMeans.predict(data_test)
    print('mini batch kmeans done')
    compare_class(predict, label_test)
    if args.create_mean:
        create_images_from_rows('mbkm', MKMeans.cluster_centers_)

def kmeans(data_train, data_test, label_train, label_test, args):
    print('kmeans')
    switch_choice = {'k': lambda: 'k-means++', 'r': lambda: 'random',
                     'm': lambda: mean_image(label_train, data_train)}
    kmeans = KMeans(n_clusters=10, random_state=0, n_init=10,
                    init=switch_choice[args.init]()).fit(data_train)    
    predicted = kmeans.predict(data_test)
    print('kmeans done')
    compare_class(predicted, label_test)
    if args.create_mean:
        create_images_from_rows('km', kmeans.cluster_centers_)

def main():
    print('start clustering')
    args = kmeans_args()
    rand = np.random.randint(10000000)
    data_train, data_test = pre_processed_data(args, rand)
    label_train, label_test = pre_processed_label(args, rand)
    print('data loaded')
    kmeans(data_train, data_test, label_train, label_test, args)
    minibatchkmeans(data_train, data_test, label_train, label_test, args)
    featureagglomeration(data_train, data_test, label_train, label_test, args)
    birch(data_train, data_test, label_train, label_test, args)
    spectralclustering(data_train, data_test, label_train, label_test, args)
    print('done')

def compare_class(predicted, label):
    unique_p, counts_p = np.unique(predicted, return_counts=True)
    found = dict(zip(unique_p, counts_p))
    unique_l, counts_l = np.unique(label, return_counts=True)
    label_nb = dict(zip(unique_l, counts_l))
    print('found: ', found)
    print('label: ', label_nb)
    matrix_confusion(label, predicted, unique_l)
    # for j in range(0, len(unique_l)):
    #     predicted = (predicted + 1) % len(unique_l)
    #     matrix_confusion(label, predicted, unique_l)

if __name__ == '__main__':
    main()

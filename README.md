# Coursework 1 

## Installation

* Clone the repository
* Create a virtualenv and activate it
* Install all the packages listed in the `requirements.txt` file (`pip install -r requirements.txt`)
* Create a `data` folder at the root of the project
* Create a `img` and a `random` folder in the `data` folder
* Add the data in the data folder
* Run the program `randomize.py` (`python randomize.py`) to create the randomized files

## Option

Every program will use by default the file `r_x_train_gr_smpl.csv` in the folder `../data/random`. The label name by default are `r_y_train_smpl(_X{0-9}).csv`.   

All the program can be run with the following options:

* `-h`: display the help

* `-col X (Y Z ...)`: use only the column given as argument. The pre-processing won't work efficiently because most of the filter need a square image, and if the number of column isn't a perfect square, the rest of the asked pre-preprocessing won't work.  
ex: `python sk_kmeans.py -col 50 51 52 53`: use only the attributes 50, 51, 52 and 53 to run naive bayes

* `-r`: randomize the file (it will use the same seed for the data file and the label files). Can be use to obtain different result on the same data set.  
ex: `python sk_kmeans.py -r`

* `-x X (Y Z ...)`: extract the column after the processing instead of before like `-col`
ex: `python sk_kmeans.py -x 50 51 52 53`: use only the attributes 50, 51, 52 and 53 to run naive bayes, but after the pre-processing being done on the whole images

* `-s X`: split the data set in train and test set. Take a number as argument. If it's a float between 0 and 1, it will cut the data set by percentage (X% for the train set). If it's an integer, the train set will be X long. The rest always go to the test set.  
ex: `python sk_means.py -s 0.7`: train set=70% | test set=30%  
ex: `python sk_means.py -s 2000`: train set=2000 | test set=total length - 2000

* `-z X`: resize by scaling the image to a square of X pixels on each side.  
ex: `python preprocess_ex.py -z 32`: the image will be scaled to 32x32 pixels

* `-p X`: apply max pooling on the image to reduce it. The matrix used will be of size (X,X).  
ex: `python preprocess_ex.py -p 2`: the image will be reduce to 24x24 images by max pooling

* `-hi`: change the histogram of each histogram by normalizing it with the histograms of all images.  

* `-fi {s,r,p,c,m,g}`: apply a filter on the image to detect the edges and outline them. The available filter are: s:Sobel, r:Roberts, p:Prewitt, c: Scharr, m:Median, g: Gaussian.  
ex: `python preprocess_ex.py -fi s`: apply sobel filter on the images

* `-eh`: equalize the histogram to even it.  

* `-bin`: binarize the image using the isodata threshold

* `-g X(t)`: apply a segmentation filter on the images. f=Felzenzwalb, w=Watershed s=Slic. Additionaly, by adding `t` like that : -g st, combine lower region under threshold  
ex: `python preprocess_ex.py -g f`: apply Felzenzwalb filter to the images
ex: `preprocess_ex.py -g wt`: apply Watershed filter and then cut the threshold

* `-d X`: will replace the default data file name by X. Need the extension

* `-l X` will replace the default label name by X. Don't provide the extension

* `-f X` will replace the default folder name by f. Don't forget the `/` at the end  
ex: `python sk_kmeans.py -d file1.csv -f ../data/ -l label_cool`: will use the data file `../data/file1.csv` and the labels files `../data/label_cool(_{0-9}).csv` instead of the default files.

## Pre-processing
### Pre-process a file
The programme `preproces_ex.py` can be run with python:
```
python preprocess_ex.py 
```
By default the file generated will be named `processed_data.csv` and the labels file `processed_data_l.csv`.

It takes a file and apply a set of filter defined by the command line argument given and output the resulting file.
### Option

* `-n` : name of the output file (without extension).  
ex: `python preprocess_ex.py -bin -n p1`: will output a file named `p1.csv`, which the binarized version of the default file.


### Generate an image
Images can also be generated to see what the different filter are. 
The program `create_image.py` can be run with python:  
```
python create_image.py
```
By default the X files generated will be named `img_{X}.ppm`. The format used is [netpbm](https://en.wikipedia.org/wiki/Netpbm_format) P2, for gray images. 
### Example
```
python preprocess_ex.py -hi -n new_hi
python preprocess_ex.py -eh -hi -d new_hi.csv -n nhhh # generate a file with all histogram equalized
```

#### How To
To avoid creating 12660 images, it it adviced to create a data set of 10 images and run the different experiment on it. It can me mixed with pre-processing the files, for example to apply twice the same filter.  
And when a good combinations of filter is found, the full process can be applied to the full data set.

#### Option
* `-m`: this option can be used to generate the average of all the image of a label. It will need a label files (default or custom).

#### Example
```
python preprocess_ex.py -n r_hi -hi # match all histogram
python preprocess_ex.py -d r_hi.csv -n r10_hi_eh_hi -hi -eh -s 10 # equalize all each histogram, rematch them all and reduce the data set to 10 images
python create_image.py -d r10_hi_eh_hi.csv # create the images without filter to see
python create_image.py -d r10_hi_eh_hi.csv -bin -n r10bin # create the image with a binary filter 
python create_image.py -d r10_hi_eh_hi.csv -fi r -bin -n r10binfir # create the image with a roberts filter and a binary filter
python preprocess_ex.py -n total_hhh_10_att -d r_hi.csv -eh -hi -x 1173 1172 1095 1047 1132 1084 1273 1321 1850 1851 1320 1368 1187 1609 1121 1073 1423 1471 1412 1363 # create a file with 10 attributes from a triple equalized/matched histograms files
```
 

## Naive bayes
### Option

* `-b {gnb,cnb,bnb,mnb}`: the type of naive bayes algorithm to be used. type of naive used, Gaussian=gnb, Bernouilli=bnb,Complement=cnb, Multinomial=mnb. Description can be found in the scikit-learn documentation. (gnb was mainly used)  
ex: `python sk_bayes.py -b gnb -bin`: use bayes on the binarized default file with the gaussian naive bayes

* `-cm`: generate the image of the mean of the label found by naive bayes

### Example
```
python sk_bayes.py -b gnb -fi r -bin # run bayes after applying roberts filter and binarizing the images
python sk_bayes.py -bnb -bin # apply bernouilli filter on binarized images
```

## K-Means

### Option
* `-i {m, r, k}`: choose the method to initialize kmeans centroids: k=kmeans++, m=mean, r=random  
ex: `python sk_kmeans.py -i m`: to use the mean of each label to initialize kmeans

* `-cm`: generate the image of the mean of the label found by kmeans


### Example
```
python preprocess_ex.py -n r_hi -hi # match all histogram
python sk_kmeans.py -cm -d r_hi.csv -bin # run sk_kmeans and generate the mean of the label as image on a binarised file named total_hi.csv

python sk_kmeans.py -d r_hi.csv -fi p -c e6 -bin # run a not so bad kmeans

python fs_kmeans.py # run a hand made kmeans
```

## NEAT algorithm

### Constructor

Generation(population, data, label)
* `population`: define the the number of neural network in a generation
* `data`: the data to train
* `label`: the label to use with the data

### Functions

* `cleanFitness()`: reset the fitness of all neural networks
* `run_generation()`: use all data to calculate the fitness of all neural networks
* `reproduce()`: copies the top 10 neural networks in the entire population after mutates the ones copied

## CSV to ARFF
This script convert a `csv` file to `arff` by mixing a label file and a data file together.  
To simplify the process the program don't other algorithm than `-d`, `-f` and `-l` to change the target file and the `-n` to name the output file.

By default the output file will be named `arf_y_train(_{0-9})
.arff`

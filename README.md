# Coursework 1 

## Installation

* Clone the repository
* Create a virtualenv and activate it
* Install all the packages listed in the `requirements.txt` file (`pip install -r requirements.txt`)
* Create a `data` folder at the root of the project
* Create a `img` and a `random` folder in the `data` folder
* Run the program `randomize.py` (`python randomize.py`) to create the randomized files
* 
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
```python
python preprocess_ex.py 
```
By default the file generated will be named `processed_data.csv` and the labels file `processed_data_l.csv`.
### Option


### Generate an image



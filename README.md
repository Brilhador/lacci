# Classification of Weeds and Crops at the Pixel-Level Using Convolutional Neural Networks and Data Augmentation

### Abstract

The Pixel-Level Classification of crops and weeds is an open problem in computer vision. The use of agrochemicals is necessary for effective weed control, but one of the great challenges of precision agriculture is to reduce their use while maintaining high crop yields.  Recently, automated weed control techniques based on computer vision were developed despite experiencing difficulties in creating agricultural datasets. One possible solution to the small volume of data available is Data Augmentation. This paper investigates the impact of individual data augmentation transformations on the pixel-level classification of crops and weeds when using a Deep Learning model.  It also investigates the influence of input image resolution on the classification performance and proposes a patch augmentation strategy. Results have shown that applying individual transformations can be valuable to the model, but gets outperformed by the combination of all transformations. This work also finds that higher resolution inputs can increase the classification performance when combined with augmentation techniques, and that patch augmentation can be a valuable asset when dealing with a small number of high-resolution images. The method reaches the mark of 83.44\% in Average Dice Similarity Coefficient, an increase of 19.96\% percentage points compared to the non-augmented model.

# Paper

Available [here](https://ieeexplore.ieee.org/abstract/document/9037044).

If you find this paper/code is useful in your research, please consider citing:

Bibtex:
```
@INPROCEEDINGS{9037044,
  author={A. {Brilhador} and M. {Gutoski} and L. T. {Hattori} and A. {de Souza Inácio} and A. E. {Lazzaretti} and H. S. {Lopes}},
  booktitle={2019 IEEE Latin American Conference on Computational Intelligence (LA-CCI)}, 
  title={Classification of Weeds and Crops at the Pixel-Level Using Convolutional Neural Networks and Data Augmentation}, 
  year={2019},
  pages={1-6},}
```

# Use

All data and code is subject to copyright and may only be used for non-commercial research. In case of use please cite our publication.

Contact Anderson Brilhador (andersonbrilhador@gmail.com, brilhador@utfpr.edu.br) for any questions.

# Run Code

1. Clone this repository </br>
``` git clone https://github.com/Brilhador/lacci_6th_cropweed.git ```
2. Install the requirements </br>
``` pip install -r requirements.txt ```
3. Run the **train_unet.py** file to train the network and get the results </br>
``` python train_unet.py```

### Brief description of the files:

requirements.txt -> required packages </br>
crop.py -> generates the patch augmentation </br>
dataset_cropweed.py -> loads images and annotations </br>
keras_train_functions.py -> makes data augmentation </br>
report_functions.py and report_metrics.py -> functions used in the report </br>
model_unet.py -> keras model u-net </br>
train_unet.py -> main script for running the experiments </br>

This repository has been changed and tested to run with the following package versions:

```
python==3.6.9
tensorflow-gpu==2.2.0
keras==2.3.1
opencv-python==4.2.0.34
scikit-image==0.17.2
scikit-learn==0.23.0
seaborn==0.10.1
mlxtend==0.17.2
```

Initially the code was developed in previous versions of tensorflow-gpu and keras, so there are some warnings in the code.




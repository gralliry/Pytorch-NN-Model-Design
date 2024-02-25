# Pytorch-Network-Model-Design

## Description

Many tasks of ```Pytorch``` for learning Network-Model-Design included.

Some of the tasks can be found in [kaggle.com](https://kaggle.com):

* This is just a practice project, and many of the tasks on kaggle.com don't have great results.

* In fact, there is no need to use __pytorch__ for some __machine learning__ tasks (unless you have a high understanding of
  the underlying principles of machine learning), which can lead to some less efficient work.

* If you can, try to use machine learning to complete tasks on kaggle.

I'm putting together a collection of simple and helpful tasks that I hope will help myself and others who
want to learn deep learning. (Or just for a simple deep learning task.)

Consider introducing tasks for ```Scikit-Learn```, ```TensorFlow```, etc. later. (just consider)

## Examples / Tasks

* Chinese to English
* Next Frame Prediction
* Sentiment Analysis on Movie Review
* Binary Classification with a Bank Churn Dataset
* Multi-Class Prediction of Obesity Risk
* Regression with a Mohs Hardness Dataset
* ......

## Structure

The sample files are in the ```template``` folder.

Note: 

* Since the general model parameter file is large, it will not be uploaded to __GitHub__, if the code does not
have the logic to create this folder, please create your own ```checkpoint``` folder.
* If the dataset file is small ( <5M ), it is uploaded directly to __GitHub__ in the ```dataset``` folder. Some cleaned data is also placed / generated in this folder.

```
dataset/      # folder to store the data set

checkpoint/   # folder to store model parameters

criterion.py  # custom loss function

dataset.py    # custom dataset

model.py      # model

train.py      # train the model

test.py       # test the model
```

## Contributors

* [gralliry](https://github.com/gralliry)
* zuozuo

## Contact

Email: ```aiccyxixy@163.com```

## License

```GNU General Public License v3.0```

This project has open source tasks, according to the provisions of the open source agreement, this project is open
source
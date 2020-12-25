## MNIST Dataset with Convolutional Neural Networks

![Alt Text](./media/thumbnail_video.gif)

### Dependencies 
* torch
* cv2 
* numpy

### Usage
Train the model:

```bash
$ python3 src/train_model.py
```

Evaluate the model with mouse input:

```bash
$ python3 src/evaluate_model.py
```

### Info
There are 2 pretrained models saved in /src/models. 


The first model takes inspiration [from this tutorial](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/), here is the model:


![alt text](./media/model_diagram.jpeg)


The second model adds a new convolutional layer. 


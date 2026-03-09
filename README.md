# Iris Softmax Classifier

This project implements a Neural Network with Softmax output to perform multiclass classification on the Iris dataset.

## Dataset
The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model classifies flowers into 3 species:
- Setosa
- Versicolor
- Virginica

## Technologies
- Python
- TensorFlow
- NumPy
- Scikit-learn

## Training

Run:

python train.py

This trains the neural network and saves the model.

## Prediction

Run:

python predict.py

Example output:

Predicted flower: Setosa

## Model Architecture

Input layer: 4 features  
Hidden Layer 1: 16 neurons (ReLU)  
Hidden Layer 2: 12 neurons (ReLU)  
Output Layer: 3 neurons (Softmax)

## License

MIT License
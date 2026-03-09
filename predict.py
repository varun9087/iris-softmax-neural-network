import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

species = ["Setosa", "Versicolor", "Virginica"]

# Example input
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(sample)

predicted_class = np.argmax(prediction)

print("Predicted flower:", species[predicted_class])
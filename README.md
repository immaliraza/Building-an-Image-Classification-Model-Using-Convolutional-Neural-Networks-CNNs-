# CNN for Image Classification

This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** for image classification tasks using Python and TensorFlow/Keras. The notebook includes all the necessary steps for data preprocessing, model creation, training, evaluation, and visualization of results. Below are the detailed explanations of each step included in the notebook.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [How to Run](#how-to-run)
9. [Conclusion](#conclusion)

---

## Project Overview

The goal of this project is to classify images into their respective categories using a CNN. Convolutional Neural Networks are widely used for image-related tasks due to their ability to extract spatial hierarchies in images.

This notebook:
- Preprocesses the dataset.
- Builds and compiles a CNN using Keras.
- Trains the model on the dataset.
- Evaluates the model's performance.
- Visualizes the training progress and sample predictions.

---

## Dependencies

The following Python libraries are used:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
```

Make sure to install them before running the code:

```bash
pip install tensorflow matplotlib
```

---

## Dataset

The dataset is loaded using the `ImageDataGenerator` class from Keras. It assumes the following directory structure:

```
root_directory/
|-- train/
|   |-- class_1/
|   |-- class_2/
|   |-- ...
|-- validation/
    |-- class_1/
    |-- class_2/
    |-- ...
```

The train folder contains the training images for each class, while the validation folder contains the validation images.

---

## Data Preprocessing

Data augmentation and preprocessing are performed using `ImageDataGenerator`:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'path_to_train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # Change to 'categorical' for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    'path_to_validation_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
```

---

## Model Architecture

The CNN model is built using the `Sequential` API from Keras. It includes convolutional, max-pooling, dropout, and dense layers:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Change to Dense(number_of_classes, activation='softmax') for multi-class classification
])

model.compile(
    loss='binary_crossentropy',  # Change to 'categorical_crossentropy' for multi-class classification
    optimizer='adam',
    metrics=['accuracy']
)
```

---

## Training the Model

The model is trained using the `fit` method:

```python
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)
```

---

## Evaluation

The model's performance is evaluated on the validation set:

```python
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
```

---




### Sample Predictions

You can use the trained model to make predictions on new images:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print('Prediction:', 'Class 1' if prediction[0] > 0.5 else 'Class 0')
```

---

## How to Run

1. Clone this repository.
2. Install the required dependencies:

   ```bash
   pip install tensorflow matplotlib
   ```

3. Prepare your dataset in the directory structure mentioned above.
4. Update the file paths in the notebook/code to point to your dataset.
5. Run the notebook or script to train and evaluate the model.
6. Visualize the results and test the model with new images.

---

## Conclusion

This project demonstrates a simple yet effective implementation of a CNN for image classification. By modifying the dataset, hyperparameters, or model architecture, you can adapt this pipeline for various image classification tasks.

## For more understanding visit My medium article
https://medium.com/@mraza1/building-an-image-classification-model-using-convolutional-neural-networks-cnns-604971fc0bed


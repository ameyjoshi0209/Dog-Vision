# Dog Vision 🐶🐕
Welcome to Dog Vision a Dog Breed Classification GitHub repository!📁.

This repository houses our state-of-the-art AI model designed for accurately identifying and classifying dog breeds from images. Leveraging deep learning techniques and a comprehensive dataset, our model promises to advance applications in veterinary diagnostics, pet adoption processes, and animal welfare.  
Feel free to explore our code, datasets, and documentation. We encourage collaboration and welcome your feedback as we continue to refine and enhance this impactful project.🔍🧭

Thank you for your interest and support!❤️😊<br>

## Model Summary
The Dog Breed Classification Model using MobileNetV2 which is a convolutional neural network (CNN) architecture designed for mobile and embedded vision applications. It consists of depth wise separable convolutions, which make it computationally efficient and suitable for devices with limited computational resources. This instance of the model use the Dog Breed Dataset to predict the name of the breed of the given dog image. The model uses the data from the [The Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs).
*You can find the model in model directory under Dog Vision directory*

Here are the couple of images from the dataset on which it is trained.<br><br>
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F21582481%2Fe0cffc59dc68b62d0ffee22a71e92251%2Foutput.png?generation=1721715711228779&alt=media)

## Usage
The MobileNetV2 model can be used for image classification tasks, such as identifying different dog breeds from images. Transfer learning is employed by fine-tuning the pre-trained MobileNetV2 model on a specific dataset of dog images.

The model requires the input shape (224, 224, 3) which ensures compatibility and optimal performance when using dog breed classification. Always preprocess your data to match this input shape to achieve the best results.<br><br>


*__NOTE: Although this document details the comprehensive usage of the model, for more information regarding usage and implementation of the model please refer my [Dog Breed Classification Model](https://www.kaggle.com/models/ameyjoshi0209/dog-breed-classification) on Kaggle.__*

<br>

To load the model, create the following function
```python
def load_model(model_path):
    """
    Loads a saved model from a specified path
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model
```
and finally load the model,
```python
load_full_model = load_model(
    "path/to/your/model"
)
```
Firstly we need to get the filepaths of images
```python
# Get custom file path
import os

custom_path = "your/filepath/"
custom_image_path = [custom_path + fname for fname in os.listdir(custom_path)]
custom_image_path
```
Next, we need to turn our images into data batches.
```python
custom_data = create_data_batches(custom_image_path, test_data=True)
```

Following function(s) create the data batches for images.
```python
# define batch size
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(
    X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if its training data but dosent shuffle if its validatoin data.
    Also accepts test data as input (no labels).
    """
    # If data is test data set, we probably dont have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X))
        )  # only file paths and no labels
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If data is a valid dataset, we dont need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X), tf.constant(y))  # filepaths
        )  # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch
    else:
        print("Creating training data batches")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))

        # Shuffling labels and pathnames before mapping, image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(X))

        # Create (image, label) tuples (this also turns the image path into preprocessed image)
        data = data.map(get_image_label)

        # Turn the training data into batches
        data_batch = data.batch(BATCH_SIZE)

    return data_batch
```
```python
# Create a function to return a tuple (image, label)
def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label
```
```python
# Defining image size
IMG_SIZE = 224


# Creating function to preprocess images
def process_image(image_path, img_size=IMG_SIZE):
    """
    Takes an image file path and turns the image into a Tensor
    """
    # Read image file
    image = tf.io.read_file(image_path)

    # Turn image into numerical tensor with 3 color channel(RGB)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert color channel value from 0-255 to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image
```

Finally we make predictions on given image input
```python
# Make predictions on custom data
custom_preds = load_full_model.predict(custom_data)
```

The output will be in `(number_of_images, 120)` shape format. Therefore we need to retrieve the predicted labels.

```python
# Get custom image prediction labels
custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_pred_labels
```
Also we need to convert it to string format which would be more easy to understand. Following functions come handy for the same.
```python
# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]
```
Here unique breed is an array of unique breeds
```python
array(['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier'], dtype=object)
```
<br>


## System
The Dog Breed Classification model using MobileNetV2 can be considered a standalone model designed for a specific task: classifying images of dog breeds. However, in practical applications, such a model is often part of a larger system or pipeline.

## Implementation requirements
Implementation of the Dog Breed Classification model using MobileNetV2 requires the following software, hardware, and dataset preparation steps:

### Software Requirements
**Python Environment**
* Python 3.x (preferably Python 3.6 or higher)

**Libraries and Frameworks**
* TensorFlow: Deep learning framework for building and training neural networks.
* Keras: High-level neural networks API, running on top of TensorFlow (or other backend engines).

**Additional Python Libraries**
* NumPy: Fundamental package for numerical computing with Python.
* Matplotlib: Plotting library for visualizing data and model performance.
* Sklearn: for additional functionalities, especially for tasks such as preprocessing, evaluation metrics, or further downstream processing

### Hardware Requirements
**CPU**
* Minimum: Multi-core CPU (e.g., Intel i5 or AMD Ryzen 5 series)
* Recommended: Multi-core CPU with higher clock speed for faster training

**GPU (Optional, but Recommended for Faster Training)**
* NVIDIA GPU with CUDA support (e.g., GTX 1060, RTX 2080)
* TensorFlow-GPU version for utilizing GPU acceleration

## Model Characteristics

### Model initialization

In the implementation of a Dog Breed Classification model using MobileNetV2, fine-tuning from a pre-trained model (pre-trained on ImageNet) was the taken approach. This strategy balanced model performance and computational efficiency by leveraging the knowledge encoded in the pre-trained weights and adapting it to the specific task of classifying dog breeds.

### Model stats
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer (KerasLayer)    (None, 1001)              5432713   
                                                                 
 dense (Dense)               (None, 120)               120240    
                                                                 
=================================================================
Total params: 5,552,953
Trainable params: 120,240
Non-trainable params: 5,432,713
_________________________________________________________________
```


## Data Overview  
The dataset on which the model is trained on is taken from http://vision.stanford.edu/aditya86/ImageNetDogs .

### Usage limitations

Yes, there are sensitive use cases associated with the deployment of a Dog Breed Classification model using MobileNetV2. Understanding these sensitivities and factors that could limit model performance is crucial for responsible deployment and usage. Here are key considerations:

#### 1. Data Quality
* The quality and diversity of the training data directly influence the model's ability to generalize to unseen images.
* Limited or biased training data (e.g., imbalance across dog breeds) can lead to suboptimal performance and skewed predictions.

#### 2. Model Architecture and Hyperparameters:
* The choice of model architecture (MobileNetV2 in this case) and hyperparameters (e.g., learning rate, batch size) significantly impacts model performance.
* Improper configuration or selection may hinder convergence or lead to overfitting.

## Provenance
### Source of Model  
https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-classification/2

### Dataset Information  
The Stanford Dogs Dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/) is used, containing images of 120 dog breeds. It is split into train, validation, and test sets with appropriate preprocessing for image classification tasks.

### References  
TensorFlow and Keras official documentation: https://www.tensorflow.org/ and https://keras.io/

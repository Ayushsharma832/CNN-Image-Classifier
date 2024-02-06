
# CNN-Image-Classifier

This project demonstrates a simple image classifier using Convolutional Neural Networks (CNN) with TensorFlow. The dataset used contains 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.


## Skills:

- Image Classification
- Convolutional Neural Networks
- Model Evaluation
## Libraries:

- TensorFlow
- Matplotlib
- NumPy
- Scikit-learn (for classification report)
## Project Structure

- image_classifier.py: Python script containing the CNN model and training process.

- README.md: Project overview and instructions.


## Usage

Clone the repository:

```bash
  git clone <repository_url>
  cd cnn-image-classifier
```

Run the script:

```bash
  python image_classifier.py
```


## Results

- The script loads the CIFAR-10 dataset, preprocesses the images, and trains a CNN model.
- Model evaluation is performed, displaying accuracy on testing data.
- Sample predictions are made and compared to the original labels.


## Model Architecture

- Input Layer: Conv2D (64 filters, kernel size 3x3, ReLU activation)
- MaxPooling Layer (2x2)
- Conv2D (64 filters, kernel size 3x3, ReLU activation)
- MaxPooling Layer (2x2)
- Flatten Layer
- Dense Layer (64 units, ReLU activation)
- Output Layer (10 units, Softmax activation)
## Model Training

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy
## Evaluation

The model achieves an accuracy of approximately 70% on testing data.
## Conclusion

The CNN model outperforms a simple artificial neural network, showcasing the effectiveness of CNNs for image classification.
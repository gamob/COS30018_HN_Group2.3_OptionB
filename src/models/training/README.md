

### Machine Learning Model Workflow (Task 3)
This workflow handles the training, evaluation, and saving of the Convolutional Neural Network (CNN) used to recognize individual digits.

#### 1. Dataset Acquisition & Preprocessing
The system uses the MNIST dataset for training and initial evaluation.
   Loading: Data is split into training and testing sets via `mnist.load_data()`.
   Normalization: Pixel values are scaled to a range of  to improve model convergence.
   Reshaping: Images are reshaped to (28, 28, 1) to match the input requirements of the CNN layers.



#### 2. Model Architecture (CNN)
The model is built using a Sequential approach with the following layers:
   Conv2D: Extracts image features (edges, curves, and shapes).
   MaxPooling2D: Reduces image spatial dimensions to prevent overfitting.
   Flatten: Converts 2D feature maps into a 1D vector.
   Dense (Hidden): Learns complex patterns from the extracted features.
   Dense (Output): Uses Softmax activation to provide probability scores for digits 0-9.



#### 3. Training Pipeline (`train_cnn.py`)
The training process follows these steps:
1.  Compilation: Uses the Adam optimizer and sparse_categorical_crossentropy loss function.
2.  Fitting: The model trains for 5 epochs, repeatedly comparing predictions to correct labels.
3.  Evaluation: Performance is measured against the test set, with an expected accuracy of 97–99%.
4.  Saving: The final trained model is exported as `digit_cnn_model.h5` for integration into the main system.



#### 4. Model Testing (`test_model.py`)
This script verifies the saved model's performance on unseen data:
   Loads the `.h5` model file.
   Preprocesses a test image from the MNIST test set.
   Outputs the Predicted Digit vs. the Actual Digit to verify accuracy.



#### 5. Purpose & Integration
This ML component serves as the core recognition engine for the system:
   OCR Engine: Classifies digits extracted during the Segmentation (Task 2) phase.
   Evaluation: Provides the basis for the mandatory comparison of different ML techniques required for the project report.
   GUI Integration: The saved model is called by the system's interface to provide real-time recognition of user-inputted numbers.
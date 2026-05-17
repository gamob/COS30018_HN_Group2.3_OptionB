3. Machine Learning Model (Task 3)

This is the "brain" of the operation! It takes those single digits and tries to guess what they are.

Architecture: You'll likely use Convolutional Neural Networks (CNNs), though the project encourages you to research and compare different techniques.

Training: You’ll use the MNIST dataset to teach your model what handwritten numbers look like.

Prediction: The model outputs a probability (e.g., "I'm 99% sure this squiggle is a 7!").  

What to do: This is the big one! You must research and implement different ML models (like a basic Neural Network vs. a CNN) to compare their performance. Use libraries like PyTorch, TensorFlow, or Keras.   
What to use to test: Use the MNIST dataset. Create a script that prints out the "Accuracy" and a "Confusion Matrix" to show which numbers it gets mixed up (like 4s and 9s!).   
The Goal: Find the model with the highest accuracy to be the "final" brain of the system.   
Success Metric: An accuracy score (hopefully $>95\%$!).





ML Model Skeleton Brief

1. Primary Objective
Research, experiment with, and implement multiple machine learning techniques to identify and classify individual handwritten digits.

2. Core Data & Tools
 Mandatory Dataset: Use the MNIST dataset for training and evaluating your models.
   Frameworks: Utilize industry-standard libraries such as TensorFlow, Keras, or PyTorch.
  Supporting Libraries: Leverage `numpy` and `pandas` for data manipulation.

3. Model Research & Implementation (Weeks 5-8 Focus)
   Technique Diversity: While Convolutional Neural Networks (CNNs) are highly recommended, your team is required to investigate and learn about different approaches and models to allow for comparison.
   Team Participation: Every student in the group must be directly involved in the research and implementation of these models.
   Documentation: You must provide clear evidence (in your report and via Git commits) of the various techniques your team experimented with.

4. Evaluation and Comparison (Task 4)
   Performance Metrics: Perform a comprehensive evaluation of each implemented model’s accuracy and performance.
  Comparison Evidence: You are required to produce evidence (such as performance charts or test results) to prove that your finally selected model is the best-performing one among all those implemented by your team.
   Test Cases: Models should be tested against both single-digit images and multi-digit number constructions.

5. Selection and Integration
   Final Choice: Based on your comparative research, select the single "best" model to be integrated into the final functional system.
  
  GUI Interaction: The final system must allow users to choose between applicable models (if multiple are kept) and adjust hyperparameters to see how they affect the recognition results.

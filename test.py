from main import DeepLearning
"""
Executes the best model found by comparing the R square value
"""
# TEST BEST MODEL
DeepLearning().test_model()

# -- TEST PER MODEL --
"""
Choose a model from below:
    1. LinearRegression 
    2. DeepNeuralNetwork_16 - One hidden layer of 16 neurons
    3. DeepNeuralNetwork_30_8 - Two hidden layer with 30 and 8 neurons
    4. DeepNeuralNetwork_30_16_8 - Three hidden layers with 30, 16, 8 neurons
    5. DeepNeuralNetwork_30_16_8_4 - Four hidden layers with 30, 16, 8, 4 neurons

Choose your own learning rate, epoch number, and non linear function (default is Sigmoid)
"""
# Example below
# model = DeepLearning().test_per_model(model_name="DeepNeuralNetwork_30_16_8", learning_rate=0.01, epochs=3000, nonlinear="Sigmoid")

# -- TEST PER LEARNING RATE --
"""
Choose a learning rate from below:
    1. lr = 0.1
    2. lr = 0.01
    3. lr = 0.001
    4. lr = 0.0001

The functions executes all models (Linear Regression and Deep Neural Network)
Note: Pre-determined epoch counts and non linear function for each model are set to achieve good R^2 scores.
"""
# Example below
# DeepLearning().test_per_learning_rate(lr=0.1)
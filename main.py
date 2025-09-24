import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime
from torch import no_grad

class DeepLearning:

    def __init__(self):
        # Load dataset
        self.data = pd.read_csv('cancer_reg-1.csv', encoding='latin1')
        # Mapping of activation function names to PyTorch classes
        self.activation_functions = {
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU()
        }
        # Preprocess data and split into training, validation, and test sets
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.setup()

    def setup(self):
        # Pre-processing dataset
        # Drop column "PctSomeCol18_24" as it has over 2000 records missing
        self.data.drop(columns=['PctSomeCol18_24'], inplace=True)

        # replace NaN in 'PctEmployed16_Over' and 'PctPrivateCoverageAlone' with their respective column means
        mean_PctEmployed16_Over = self.data['PctEmployed16_Over'].mean()
        self.data['PctEmployed16_Over'] = self.data['PctEmployed16_Over'].fillna(mean_PctEmployed16_Over)
        mean_PctPrivateCoverageAlone = self.data['PctPrivateCoverageAlone'].mean()
        self.data['PctPrivateCoverageAlone'] = self.data['PctPrivateCoverageAlone'].fillna(mean_PctPrivateCoverageAlone)

        # Drop 'binnedInc' and 'Geography' columns
        # 'binnedInc' is a categorical column
        # 'Geography' is not needed for training
        self.data.drop(columns=['binnedInc', 'Geography'], inplace=True)

        # Split data into Features (X) and Target (Y)
        X = self.data.drop(columns=['TARGET_deathRate'])
        Y = self.data['TARGET_deathRate']

        # Split data into Training, Validation, and Testing
        # Training : 70%; Validation : 10%; Testing : 20%
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

        # Feature scaling
        scaler = StandardScaler() # Using sklearn StandardScaler
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Convert to Pytorch tensors
        # Covert y_train, y_val, and y_test into numpy before creating tensor
        X_train = torch.tensor(X_train, dtype=torch.float)
        y_train = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)

        X_val = torch.tensor(X_val, dtype=torch.float)
        y_val = torch.tensor(y_val.values, dtype=torch.float).view(-1, 1)

        X_test = torch.tensor(X_test, dtype=torch.float)
        y_test = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def LinearRegression(self, learning_rate, epochs=1000):
        """
        Trains a simple Linear Regression model using PyTorch
        and evaluates it on the test dataset.

        Parameters:
            learning_rate: Learning rate for SGD optimizer.
            epochs: Number of training epochs.
        """

        # Initialize the Linear Regression model
        model = nn.Linear(self.X_train.shape[1], 1)

        # Define optimizer (Stochastic Gradient Descent) and loss function (Mean Squared Error)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Lists to store training and validation loss for plotting
        train_loss_list = []
        val_loss_list = []

        print("Starting Training")

        # - Training Phase -
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            optimizer.zero_grad()  # Clear previous gradients
            y_pred = model(self.X_train)  # Forward pass on training data
            loss = criterion(y_pred, self.y_train)  # Compute training loss
            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model weights

            # - Validation Phase -
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for validation
                y_pred_val = model(self.X_val)
                val_loss = criterion(y_pred_val, self.y_val)

            # Save losses for plotting
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            # Print progress every 500 epochs
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Training Loss {loss.item():.4f}, Validation Loss {val_loss.item():.4f}")

        # Plot training and validation loss
        self.plot(model="Linear Regression", lr=learning_rate, train_loss=train_loss_list, val_loss=val_loss_list)

        # - Testing Phase -
        model.eval()
        with torch.no_grad():  # Disable gradients for test set
            y_pred_test = model(self.X_test)

        # Compute R squared score on test set
        r2 = r2_score(self.y_test.numpy(), y_pred_test.numpy())
        print(f"Linear Regression - Learning Rate = {learning_rate} - R squared on test set: {r2:.4f}")

        return model

    def DeepNeuralNetwork_16(self, learning_rate, nonlinear="Sigmoid", epochs=1000):
        """
        Trains a simple Deep Neural Network with 1 hidden layer of 16 neurons.
        Uses the specified activation function and evaluates the model on the test set.

        Parameters:
            learning_rate: Learning rate for SGD optimizer.
            nonlinear: Activation function to use (default "Sigmoid").
            epochs: Number of training epochs.
        """

        # - Define the network -
        # One hidden layer with 16 neurons and user-specified activation
        class DNN16(nn.Module):
            def __init__(self, input_dim, activation):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 16),  # Input layer -> hidden layer
                    activation,  # Activation function
                    nn.Linear(16, 1)  # Hidden layer -> output layer
                )

            def forward(self, x):
                return self.net(x)

        # Choose activation function from our dictionary
        activation = self.activation_functions.get(nonlinear, nn.Sigmoid())
        model = DNN16(self.X_train.shape[1], activation)

        # Setup optimizer and loss criterion
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() # Mean Squared Error

        # Lists to track training and validation loss for plotting
        train_loss_list = []
        val_loss_list = []

        print("Starting Training")

        # - Training Phase -
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            optimizer.zero_grad()  # Clear gradients
            y_pred = model(self.X_train)  # Forward pass
            loss = criterion(y_pred, self.y_train)  # Compute training loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # - Validation Phase -
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No gradient computation
                y_pred_val = model(self.X_val)
                val_loss = criterion(y_pred_val, self.y_val)

            # Store loss values for plotting
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Training Loss {loss.item():.4f}, Validation Loss {val_loss.item():.4f}")

        # Plot training and validation loss
        self.plot(model="DN_16", lr=learning_rate, train_loss=train_loss_list, val_loss=val_loss_list)

        # - Testing Phase -
        model.eval()
        with torch.no_grad():
            y_pred_test = model(self.X_test)

        # Compute R squared score on test set
        r2 = r2_score(self.y_test.numpy(), y_pred_test.numpy())
        print(f"DNN_16 - Learning Rate = {learning_rate} - R squared on test set: {r2:.4f}")

        return model

    def DeepNeuralNetwork_30_8(self, learning_rate, nonlinear = "Sigmoid", epochs = 1000):
        """
        Trains a Deep Neural Network with 2 hidden layers (30 and 8 neurons)
        including dropout to reduce overfitting.

        Parameters:
            learning_rate: Learning rate for SGD optimizer.
            nonlinear: Activation function to use (default "Sigmoid").
            epochs: Number of training epochs.
        """

        # - Define the network -
        # Two hidden layer with 30 and 8 neurons and user-specified activation
        class DNN30_8(nn.Module):
            def __init__(self, input_dim, activation):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 30),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(30, 8),
                    activation,
                    nn.Linear(8, 1)
                )
            def forward(self, x):
                return self.net(x)

        # Choose activation function from our dictionary
        activation = self.activation_functions.get(nonlinear, nn.Sigmoid())
        model = DNN30_8(self.X_train.shape[1], activation)
        # Setup optimizer and loss criterion
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        num_epochs = epochs
        # Lists to track training and validation loss for plotting
        train_loss_list = []
        val_loss_list = []
        print("Starting Training")
        # - Training Phase -
        for epoch in range(num_epochs):

            model.train()  # Set model to training mode
            optimizer.zero_grad()  # Clear gradients
            y_pred = model(self.X_train)  # Forward pass
            loss = criterion(y_pred, self.y_train)  # Compute training loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # - Validation Phase -
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No gradient computation
                y_pred_val = model(self.X_val)
                val_loss = criterion(y_pred_val, self.y_val)

            # Store loss values for plotting
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            # Print progress every 5000 epochs
            if epoch % 5000 == 0:
                print(f"Epoch {epoch}: Training Loss {loss.item():.4f}, Validation Loss {val_loss.item():.4f}")

        # Plot the graph of training loss and validation loss
        self.plot(model="DN_30_8", lr=learning_rate, train_loss=train_loss_list, val_loss=val_loss_list)
        # - Testing Phase -
        model.eval()
        with torch.no_grad(): # No gradiant computation on test set
            y_pred_test = model(self.X_test)

        # Compute R squared
        r2 = r2_score(self.y_test.numpy(), y_pred_test.numpy())
        print(f"DNN_30_8 - Learning Rate = {learning_rate} - R squared on test set: {r2:.4f}")

        return model

    def DeepNeuralNetwork_30_16_8(self, learning_rate, nonlinear = "Sigmoid", epochs = 1000):
        """
        Trains a Deep Neural Network with 3 hidden layers: 30 -> 16 -> 8 neurons
        with dropout and user-specified activation.

        Parameters:
            learning_rate: Learning rate for SGD optimizer.
            nonlinear: Activation function to use (default "Sigmoid").
            epochs: Number of training epochs.
        """
        # - Define the network -
        # Three hidden layer with 30, 16, and 8 neurons and user-specified activation
        class DNN30_16_8(nn.Module):

            def __init__(self, input_dim, activation):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 30),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(30, 16),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(16, 8),
                    activation,
                    nn.Linear(8, 1)
                )

            def forward(self, x):
                return self.net(x)

        # Use non-linear function as requested by the user
        activation = self.activation_functions.get(nonlinear, nn.Sigmoid())
        model = DNN30_16_8(self.X_train.shape[1], activation)

        # Setup optimizer and criterion
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() # Mean Squared Error

        num_epochs = epochs
        # Store a list of train loss and validation loss for plotting
        train_loss_list = []
        val_loss_list = []
        print("Starting Training")
        # - Training Phase -
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            optimizer.zero_grad()  # Clear gradients
            y_pred = model(self.X_train)  # Forward pass
            loss = criterion(y_pred, self.y_train)  # Compute training loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # - Validation Phase -
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No gradient computation
                y_pred_val = model(self.X_val)
                val_loss = criterion(y_pred_val, self.y_val)

            # Store loss values for plotting
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Training Loss {loss.item():.4f}, Validation Loss {val_loss.item():.4f}")

        # Plot the graph
        self.plot(model="DN_30_16_8", lr=learning_rate, train_loss=train_loss_list, val_loss=val_loss_list)
        model.eval()
        with torch.no_grad(): # No gradient computation
            y_pred_test = model(self.X_test)

        # Compute R square
        r2 = r2_score(self.y_test.numpy(), y_pred_test.numpy())
        print(f"DNN_30_16_8 - Learning Rate = {learning_rate} - R squared on test set: {r2:.4f}")

        return model

    def DeepNeuralNetwork_30_16_8_4(self, learning_rate, nonlinear = "Tanh", epochs = 1000):
        """
        Trains a Deep Neural Network with 4 hidden layers: 30 -> 16 -> 8 -> 4 neurons
        with dropout and user-specified activation function.

        Parameters:
            learning_rate: Learning rate for SGD optimizer.
            nonlinear: Activation function to use (default "Sigmoid").
            epochs: Number of training epochs.
        """
        # - Define the network -
        # Four hidden layer with 30, 16, 8, and 4 neurons and user-specified activation
        class DNN30_16_8_4(nn.Module):

            # Constructor for the Neural Network
            def __init__(self, input_dim, activation):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 30),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(30, 16),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(16, 8),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(8, 4),
                    activation,
                    nn.Linear(4, 1)
                )

            def forward(self, x):
                return self.net(x)

        # Use non-linear function as requested by the user
        activation = self.activation_functions.get(nonlinear, nn.Sigmoid())
        model = DNN30_16_8_4(self.X_train.shape[1], activation)

        # Setup optimizer and criterion
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        num_epochs = epochs
        # Store a list of train loss and validation loss for plotting
        train_loss_list = []
        val_loss_list = []
        print("Starting Training")
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(self.X_train)
            loss = criterion(y_pred, self.y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            # Validation Phase -
            with torch.no_grad():
                y_pred_val = model(self.X_val)
                val_loss = criterion(y_pred_val, self.y_val)

            # Print progress every 5000 epochs
            if epoch % 5000 == 0:
                print(f"Epoch {epoch}: Training Loss {loss.item():.4f}, Validation Loss {val_loss.item():.4f}")
            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

        self.plot(model="DN_30_16_8_4", lr=learning_rate, train_loss=train_loss_list, val_loss=val_loss_list)
        model.eval()
        with torch.no_grad():
            y_pred_test = model(self.X_test)

        # Compute R square
        r2 = r2_score(self.y_test.numpy(), y_pred_test.numpy())
        print(f"DNN_30_16_8_4 - Learning Rate = {learning_rate} - R squared on test set: {r2:.4f}")

        return model

    def plot(self, model, lr, train_loss, val_loss):
        """
        Plots training and validation loss curves for a given model and learning rate.

        Parameters:
            model: Name of the model (used for labeling the plot and saving).
            lr: Learning rate used (used for labeling and saving).
            train_loss: List of training loss values for each epoch.
            val_loss: List of validation loss values for each epoch.
        """

        plt.figure(figsize=(8, 6))
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label="Training Set", color="blue")
        plt.plot(epochs, val_loss, label="Validation Set", color="green")
        plt.title("Model Loss")
        plt.xlabel("epoch")
        plt.ylabel("mean squared error loss")
        plt.legend()
        # Save the plot as a PNG file in the image folder
        plt.savefig(f"images/Model_{model}_LR_{lr}.png")

    def test_per_model(self, learning_rate, model_name, epochs, nonlinear="Sigmoid"):
        """
        Runs a specific model training based on the model name.

        Parameters:
            learning_rate: Learning rate for training.
            model_name: Name of the model to train.
            epochs: Number of epochs for training.
            nonlinear: Activation function to use in DNN models.
        """

        if model_name == "LinearRegression":
            return self.LinearRegression(learning_rate=learning_rate, epochs=epochs)
        elif model_name == "DeepNeuralNetwork_16":
            return self.DeepNeuralNetwork_16(learning_rate=learning_rate, nonlinear= nonlinear, epochs=epochs)
        elif model_name == "DeepNeuralNetwork_30_8":
            return self.DeepNeuralNetwork_30_8(learning_rate=learning_rate, nonlinear= nonlinear, epochs=epochs)
        elif model_name == "DeepNeuralNetwork_30_16_8":
            return self.DeepNeuralNetwork_30_16_8(learning_rate=learning_rate, nonlinear= nonlinear, epochs=epochs)
        elif model_name == "DeepNeuralNetwork_30_16_8_4":
            return self.DeepNeuralNetwork_30_16_8_4(learning_rate=learning_rate, nonlinear= nonlinear, epochs=epochs)

    def test_per_learning_rate(self, lr):
        """
        Runs training for all models using a specific learning rate.
        Pre-determined epoch counts for each model are set to achieve decent R^2 scores.

        Parameters:
            lr: Learning rate for this test batch.
        """

        if lr == 0.1:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Starting Training")
            self.LinearRegression(learning_rate=0.1, epochs=700) # R2 = 0.5001
            self.DeepNeuralNetwork_16(learning_rate = 0.1, nonlinear="Sigmoid", epochs = 350) # R2 = 0.4250
            self.DeepNeuralNetwork_30_8(learning_rate = 0.1, nonlinear="Tanh", epochs = 180) # R2 = 0.3376
            self.DeepNeuralNetwork_30_16_8(learning_rate = 0.1, nonlinear="Tanh", epochs = 200) # R2 = 0.2306
            self.DeepNeuralNetwork_30_16_8_4(learning_rate = 0.1, nonlinear="Tanh", epochs = 90) # R2 = 0.1985

        elif lr == 0.01:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Starting Training")
            self.LinearRegression(learning_rate=0.01, epochs=700) # R2 = 0.4977
            self.DeepNeuralNetwork_16(learning_rate = 0.01, nonlinear="Sigmoid", epochs = 1000) # R2 = 0.5058
            self.DeepNeuralNetwork_30_8(learning_rate = 0.01, nonlinear="Sigmoid", epochs = 900) # R2 = 0.5121
            self.DeepNeuralNetwork_30_16_8(learning_rate = 0.01, nonlinear="Sigmoid", epochs = 1000) # R2 = 0.5190
            self.DeepNeuralNetwork_30_16_8_4(learning_rate = 0.01, nonlinear="Tanh", epochs = 500) # R2 = 0.4251

        elif lr == 0.001:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Starting Training")
            self.LinearRegression(learning_rate=0.001, epochs=3000) # R2 = 0.4961
            self.DeepNeuralNetwork_16(learning_rate = 0.001, nonlinear="Sigmoid", epochs = 6000) # R2 = 0.5195
            self.DeepNeuralNetwork_30_8(learning_rate=0.001, nonlinear="Sigmoid", epochs = 3800) # R2 = 0.5031
            self.DeepNeuralNetwork_30_16_8(learning_rate=0.001, nonlinear="Sigmoid", epochs =4000) # R2 = 0.4614
            self.DeepNeuralNetwork_30_16_8_4(learning_rate=0.001, nonlinear="Tanh", epochs =2500) # R2 = 0.4920

        elif lr == 0.0001:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Starting Training")
            self.LinearRegression(learning_rate=0.0001, epochs=25000) # R2 = 0.4939
            self.DeepNeuralNetwork_16(learning_rate = 0.0001, nonlinear="Sigmoid", epochs = 7500) # R2 = 0.4614
            self.DeepNeuralNetwork_30_8(learning_rate = 0.0001, nonlinear="Sigmoid", epochs = 30000) # R2 = 0.4936
            self.DeepNeuralNetwork_30_16_8(learning_rate = 0.0001, nonlinear="Tanh", epochs = 6000) # R2 = 0.4162
            self.DeepNeuralNetwork_30_16_8_4(learning_rate = 0.0001, nonlinear="Tanh", epochs = 20000) # R2 = 0.4573

        else:
            print("Choose a valid learning rate.")

    def test_model(self):
        """
            Test the best model found. The model has already been trained under the following parameters
            - DNN30_16_8: learning rate = 0.01, epochs = 3000, nonlinear = Sigmoid

            Prints the R squared value found on the test set
        """

        # Recreating the same architecture
        class DNN30_16_8(nn.Module):
            def __init__(self, input_dim, activation):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 30),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(30, 16),
                    activation,
                    nn.Dropout(p=0.3),
                    nn.Linear(16, 8),
                    activation,
                    nn.Linear(8, 1)
                )
            def forward(self, x):
                return self.net(x)

        activation = nn.Sigmoid()
        input_dim = self.X_train.shape[1]

        # Initialize model
        model_loaded = DNN30_16_8(input_dim, activation)
        # Load weights
        model_loaded.load_state_dict(torch.load("DeepNeuralNetwork_30_16_8.pt"))
        model_loaded.eval()

        # Run predictions on test set
        with torch.no_grad():
            y_pred_test = model_loaded(self.X_test)

        # Calculate R square
        r2 = r2_score(self.y_test.numpy(), y_pred_test.numpy())
        print(f"DNN_30_16_8 (BEST MODEL FOUND) - R squared on test set: {r2:.4f}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %matplotlib inline

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
  # Input layer (4 features of the flower) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # output (3 classes of iris flowers)
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1).cuda(0)
    self.fc2 = nn.Linear(h1, h2).cuda(0)
    self.out = nn.Linear(h2, out_features).cuda(0)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x
  
print(torch.cuda.is_available()) # This tells me if CUDA-enabled GPU is detected and PyTorch is configured to use it
print(torch.cuda.device_count()) # Returns the available GPUs

if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # print(stats)

        # main code
        # Pick a manual seed for randomization
        torch.cuda.manual_seed(41)
        # Create an instance of model
        model = Model().cuda(0)

        url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
        my_df = pd.read_csv(url)


        my_df.tail() # This will display the last 5 rows

        # Change last column from strings to integers
        my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
        my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
        my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

        # Train Test Split!  Set X, y
        X = my_df.drop('variety', axis=1)
        y = my_df['variety']

        # Convert these to numpy arrays
        X = X.values
        y = y.values

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

        # Convert X features to float tensors
        X_train = torch.FloatTensor(X_train).cuda(0)
        X_test = torch.FloatTensor(X_test).cuda(0)

        # Convert y labels to tensors long
        y_train = torch.LongTensor(y_train).cuda(0)
        y_test = torch.LongTensor(y_test).cuda(0)
        
        # Set the criterion of model to measure the error, how far off the predictions are from the data
        criterion = nn.CrossEntropyLoss().cuda(0)
        # Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train our model!
        # Epochs? (one run thru all the training data in our network)
        epochs = 100
        losses = []
        for i in range(epochs):
        # Go forward and get a prediction
            y_pred = model.forward(X_train) # Get predicted results

            # Measure the loss/error, gonna be high at first
            loss = criterion(y_pred, y_train) # predicted values vs the y_train

            # Keep Track of our losses
            lossCPU = loss.cpu()
            losses.append(lossCPU.detach().numpy())

            # print every 10 epoch
            if i % 10 == 0:
                print(f'Epoch: {i} and loss: {loss}')

            # Do some back propagation: take the error rate of forward propagation and feed it back
            # thru the network to fine tune the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        stats = torch.cuda.memory_stats()
        
        # Graph it out!
        plt.plot(range(epochs), losses)
        plt.ylabel("loss/error")
        plt.xlabel('Epoch')
        plt.show()

        # Evaluate Model on Test Data Set (validate model on test set)
        with torch.no_grad():  # Basically turn off back propogation
          y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be predictions
          loss = criterion(y_eval, y_test) # Find the loss or error

        correct = 0
        with torch.no_grad():
          for i, data in enumerate(X_test):
            y_val = model.forward(data)

            if y_test[i] == 0:
              x = "Setosa"
            elif y_test[i] == 1:
              x = 'Versicolor'
            else:
              x = 'Virginica'


            # Will tell us what type of flower class our network thinks it is
            print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

            # Correct or not
            if y_val.argmax().item() == y_test[i]:
              correct +=1

        print(f'We got {correct} correct!')

        stats = torch.cuda.memory_stats()

        """ new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
        with torch.no_grad():
          print(model(new_iris))

        newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

        with torch.no_grad():
          print(model(newer_iris))

        # Save our NN Model
        torch.save(model.state_dict(), 'my_really_awesome_iris_model.pt')

        # Load the Saved Model
        new_model = Model()
        new_model.load_state_dict(torch.load('my_really_awesome_iris_model.pt'))

        # Make sure it loaded correctly
        new_model.eval() """
        # end main code
        
        
        """ # Returns the maximum GPU memory (in bytes) that was ever allocated during the current session
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
        # Returns the maximum GPU memory (in bytes) that was ever cached during the current session
        print(f"Max memory cached: {torch.cuda.max_memory_reserved() / (1024**2):.2f} MB") """


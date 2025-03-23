# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![WhatsApp Image 2025-03-23 at 20 21 03_1702358c](https://github.com/user-attachments/assets/be9ac640-dbf8-4524-abee-3b82acf151b8)


## DESIGN STEPS

### Step 1: 
Import necessary libraries and load the dataset.

### Step 2: 
Encode categorical variables and normalize numerical features.

### Step 3: 
Split the dataset into training and testing subsets.

### Step 4: 
Design a multi-layer neural network with appropriate activation functions.

### Step 5: 
Train the model using an optimizer and loss function.

### Step 6: 
Evaluate the model and generate a confusion matrix.

### Step 7: 
Use the trained model to classify new data samples.

### Step 8: 
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: Midhun S
### Register Number: 212223240087

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information
![image](https://github.com/user-attachments/assets/6af8fd70-076a-4bb5-9f3f-bc65d857ab1d)

<br><br><br><br>
## OUTPUT



### Confusion Matrix
![image](https://github.com/user-attachments/assets/4b1e2aac-9231-4ac6-9f09-f1fbe76f8420)


### Classification Report
![image](https://github.com/user-attachments/assets/1a7da670-7b89-4646-aca3-e8991c20c8a8)



### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/8740e4c1-17e7-48b0-adef-0a7b3a04250d)



## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.

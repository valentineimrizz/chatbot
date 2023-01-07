import json
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
import string

# Load the data
data = []
for file in ["ehealthforumQAs.json", "questionDoctorQAs.json", "icliniqQAs.json", "webmdQAs.json"]:
    with open(f"medical-question-answer-data/{file}") as f:
        data.extend(json.load(f))
df = pd.DataFrame(data)
df = pd.json_normalize(data)

df = df.drop_duplicates(['question', 'answer'])
df = df.dropna(subset=['question', 'answer'])
# Select the relevant columns
questions = df['question']
answers = df['answer']

import re

# Remove any non-alphabetic characters
questions = questions.apply(lambda x: re.sub('[^a-zA-Z\s]', '', x))






from transformers import BertModel, BertTokenizer

# Download a BioBERT model and its associated vocabulary
model = BertModel.from_pretrained('monologg/biobert_v1.1_pubmed')
tokenizer = BertTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
# Tokenize the data
questions_tokens = questions.apply(lambda x: tokenizer.tokenize(x))
answers_tokens = answers.apply(lambda x: tokenizer.tokenize(x))
# Convert the data to tensors
questions_tensor = questions_tokens.apply(lambda x: torch.tensor(tokenizer.convert_tokens_to_ids(x)).unsqueeze(0))
answers_tensor = answers_tokens.apply(lambda x: torch.tensor(tokenizer.convert_tokens_to_ids(x)).unsqueeze(0))
# Split the data into training and validation sets
train_questions, val_questions, train_answers, val_answers = train_test_split(questions_tensor, answers_tensor, test_size=0.2)
# Rename the size variable
size_variable = 10
# Create a TensorDataset from the training and validation sets
train_data = torch.utils.data.TensorDataset(train_questions, train_answers)
val_data = torch.utils.data.TensorDataset(val_questions, val_answers)
# Create a DataLoader from the TensorDataset
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)
# Create a classification model on top of BioBERT
class BioBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BioBERTClassifier, self).__init__()
        self.num_classes = num_classes
        self.bert = model
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        return output
# Define a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# Define the number of classes in the classification task
num_classes = 2

# Instantiate the BioBERTClassifier model
model = BioBERTClassifier(num_classes)

# Set the model to training mode
model.train()

# Define the number of epochs to train the model
num_epochs = 10

# Train the model for a given number of epochs
for epoch in range(num_epochs):
    # Loop over the training dataloader
    for i, (questions, answers) in enumerate(train_dataloader):
        # Move the input and target tensors to the GPU
        questions = questions.to(device)
        answers = answers.to(device)

        # Forward pass
        output = model(questions, attention_mask)
        loss = loss_fn(output, answers)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Print the loss and accuracy for each epoch
    print("Epoch: {}/{} | Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))



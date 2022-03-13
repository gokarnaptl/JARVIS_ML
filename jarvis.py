import random
import json
import torch
from Brain import NeuralNet
from NeuralNetwork import bag_of_words ,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", 'r') as json_data:
	intents = json.load(json_data)

FILE = "TrainData.ptj"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["output_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#----------------

Name = "Jarvis"
from Listen import Listen
def Main():
	sentence = Listen
	result = str(sentence)
# Main()
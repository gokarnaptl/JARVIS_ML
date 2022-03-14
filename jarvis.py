from http.client import responses
import random
import json
import torch
with open("intents.json", 'r') as json_data:
	intents = json.load(json_data)

FILE = "TrainData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
model.eval()

#----------------
Name = "Jarvis"
from Listen import Listen
from Speak import Say

def Main():

	sentence = Listen()

	if sentence == "bye":
		exit()
	
	sentence = tokenize(sentence)
	X = bag_of_words(sentence,all_words)
	X = X.reshape(1,X.shape[0])
	X = torch.from_numpy(X).to(device)

	prob = probs[0][predicted.item()]

	if prob.item() > 0.75:
		for intent in intents['intents']:
			for intent in intents['intends']:
				if tag == intent["tag"]:
					reply = random.choice(intent["responses"])
					Say(reply)

while True:
	Main()
				# speak(reply)

# Main()
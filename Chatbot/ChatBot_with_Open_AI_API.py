"""
This code generates a chatbot based on a open API. 
API is generated from open AI site, and save in local txt file. It is not upload to the git for safety

We use an API key, which is generated from open AI site, and save in a local txt file. It is not upload to the git for safety.
First, we create a virtual python environment and install the openai package. we use such virutual environemnt to sotre our secret keys
(Use [pip install virtualenv , virtualenv env] to create a virtual environment. Then use [source env/scripts/activate] to activate the environment. 
add requriemnts.txt file with the content "openai" and "python-dotenv" packages. then run pip install -r requirements.txt to install those packaes

"""

#import libraries
from openai import OpenAI
#to manage API key as a local enviornment variable we need OS library
import os
#load openai key from .env file, first import the library to load env variables
from dotenv import load_dotenv
#then load env variables from .env file
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

#create a client to connect to the API
client = OpenAI()
while True:
    question=input("User:  ")#get the user input to send to the model. We can use this input to generate a response from the model.
    if any(word in question.lower() for word in ["exit", "quit", "bye"]): #we can use this input to exit the chatbot. We can also use this input to end the conversation with the model.
        print("Goodbye!")
        break
    else:
        response=client.chat.completions.create(
            model="gpt-3.5-turbo",#we select a model to use, there are many models available, we can select the one that best suits our needs. We can also fine-tune the model to improve its performance on our specific task.
            max_tokens=100,#we can specify the maximum number of tokens to generate in the response. This is useful to prevent the model from generating too much text and to control the cost of using the API.
            n=1,#we can specify the number of responses to generate. This is useful to get multiple responses for the same input and to compare them.
            temperature=0.5,#we can specify the temperature to control the randomness of the response. A higher temperature will result in more random responses, while a lower temperature will result in more deterministic responses.
            messages=[
            {"role": "user", "content": question},#we can specify the messages to send to the model. The messages should be in the format of a list of dictionaries, where each dictionary represents a message with a role (user or assistant) and content (the input text from the user).
        ],
        )

        for choice in response.choices:
            print(f"AI: {choice.message.content}")
    
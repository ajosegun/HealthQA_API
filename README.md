# HealthQA_API

HealthQA_API is a Github repository that contains code for a RESTful API built with FastAPI and deployed on Vercel. The API uses the sentence-transformers package with the all-MiniLM-L6-v2 model for question and answer retrieval.

HealthAQ is a medical chatbot that utilizes Natural Language Processing to provide responses to user medical questions. Built on pre-trained sentence similarity model from the HuggingFace Repository, the bot has been trained on a variety of medical queries sourced from websites such as WebMD and questionDoctor, with a total of roughly 30,000 back-and-forth conversations. 

It is advanced compared to traditional conversational chatbots as it generates its own replies by analyzing the context of the user's question.

Unlike traditional methods where the chatbot relies on pre-stored replies for a set of expected questions, HealthAQ generates its own responses based on the context of the user's query.

# Installation:

To use the API, you can follow these steps:

Clone the repository by running the command git clone https://github.com/ajosegun/HealthQA_API.git

Navigate to the cloned directory using the command line

Install the required packages by running the command pip install -r requirements.txt

Start the server by running the command uvicorn main:app --reload

# Conclusion:

HealthQA_API is a powerful tool for managing health-related questions and answers. It provides an easy-to-use RESTful API built with FastAPI and deployed on Vercel. The API uses the sentence-transformers package with the all-MiniLM-L6-v2 model for question and answer retrieval, which ensures accurate and relevant search results. 

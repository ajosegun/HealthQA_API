import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Path
from sentence_transformers import SentenceTransformer, util
import io
import os
import pickle
import requests

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def download_pkl_from_drive(drive_url: str, filename: str) -> None:
    """Download a PKL file from a public Google Drive link.

    Args:
        drive_url (str): The public Google Drive URL of the file.
        filename (str): The local file name to save the PKL file as.
    """
    # Get the file ID from the Google Drive URL
    file_id = drive_url.split('/')[-2]
    
    # Construct the file download URL
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Download the file contents as a byte stream
    response = requests.get(download_url)
    content = response.content
    
    # Deserialize the byte stream as a PKL object
    pkl_object = pickle.loads(content)
    pkl_object.to_pickle(filename)

link_to_dataset_on_drive = 'df_health_ds.pkl'

if os.path.exists(link_to_dataset_on_drive) == False:
    print("Attempting to download pkl file")
    drive_url = "https://drive.google.com/file/d/1xrVoWwTaCW6AqhTxFjOvI42ZnwnhVjbJ/view?usp=sharing"
    filename = "df_health_ds.pkl"
    download_pkl_from_drive(drive_url, filename)

df_health_ds = pd.read_pickle(link_to_dataset_on_drive)
print("PKL file loaded")

def single_answer(user_question: str, top_k: int = 1) ->list:
  user_question = user_question.strip()

  if user_question == '':
    return 'Please enter a valid question'

  user_embeddings = encoder.encode(user_question, convert_to_tensor=True)

  #Compute cosine-similarities for each sentence with each other sentence
  questions_embeddings = df_health_ds['questions_embeddings'].tolist()
  hits = util.semantic_search(user_embeddings, questions_embeddings, top_k=top_k)[0]

  similar_questions, similar_answer, score = [], [], []
  
  for hit in hits:
    
    the_id = hit['corpus_id']
    the_score = hit['score']
    
    similar_questions.append(df_health_ds['questions'].tolist()[the_id])
    ### Add question answering to shorten the answer
    similar_answer.append(df_health_ds['answers'].tolist()[the_id].replace('--> <link>', ''))
    score.append("{:.4f}".format(the_score))

  return similar_questions, similar_answer, score


app = FastAPI(
    title="Health Question and Answering",
    description="A simple API that use Sentence Transformers to answer a health related question.",
    version="0.1",
)

@app.get("/answer")
def get_single_answer(question: str):
    """
    A simple function that receive a question and returns the top answer.
    :param question:
    :return: similar_questions, similar_answer, Probability
    """
    if question.strip() == "":
       return {"Error", "Question cannot be empty"}
    
    similar_questions, similar_answer, score = single_answer(question)

    # show results
    result = {"Similar Question": similar_questions[0], 
              "Answer": similar_answer[0], 
              "Probability": score[0],
              "Error": ""}
    
    return result

@app.get("/answer-top")
def get_single_answer(question: str, 
                      top_k: int):
    """
    A simple function that receive a question and returns the top answer.
    :param question:
    :return: similar_questions, similar_answer, Probability
    """
    if question.strip() == "":
       return {"Error", "Question cannot be empty"}
    
    similar_questions, similar_answer, score = single_answer(question, top_k)

    # show results
    result = {"Similar Questions": similar_questions, 
              "Similar Answers": similar_answer, 
              "Probability": score,
              "Error": ""}
    
    return result


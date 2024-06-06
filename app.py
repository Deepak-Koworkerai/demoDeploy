from flask import Flask, request, jsonify, render_template
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import time
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from google.api_core.exceptions import DeadlineExceeded
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

embeddings= model_norm


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def batch_iterable(iterable, batch_size):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def get_vector_store(text_chunks, batch_size=10):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-002", google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    Context:\n {context}?\n
    Question: \n{question}\n
    Note: answer the context relevantly and adjust to the tone and way of the user, and If the question is related to some general topic or a conservational chat, try to answer as human like if the question is "tell me something you know ", "is this right time to talk", try to answer it
    if the answer is not in provided context and if you think the question is personal or unknown or common just say, "Oh no!, I am not really aware of it, I shall ask him and let you know later!!", don't provide the wrong answer\n\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-002", google_api_key='AIzaSyBg9Hq7avlD4iX94pnU9ce6YwT1X5LPeVc')
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

@app.route('/')
def index():
    return render_template('index.html')

def vectorize():
    text_dir = os.getcwd()
    text_files = [os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith('.txt')]

    if text_files:
        raw_text = []
        for text_file in text_files:
            file_text = read_text_file(text_file)
            raw_text.append(file_text)

        raw_text = '\n'.join(raw_text)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return "Vectorized the documents successfully!"
    else:
        return "No text files found in the specified directory."

vectorize()

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    response = user_input(user_question)
    return jsonify({'response': prettify_text(response)})

def prettify_text(text):
    prettified = text.replace('\n', '<br>')
    prettified = prettified.replace('**', '<b>').replace('*', '<li>')
    prettified = prettified.replace('<b>', '</b>', 1)  # Ensure to close the first bold tag correctly
    return prettified
    
if __name__ == '__main__':
    from app import app
    app.run(debug=False, host='0.0.0.0')


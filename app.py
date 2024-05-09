from flask import Flask, request, jsonify
import textwrap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import webbrowser
from IPython.display import Markdown
# from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
import requests

app = Flask(__name__)
# run_with_ngrok(app)
ngrok.set_auth_token("2Z2bcY1mV3ORh2aeD2OBM6rebME_7jo2YNGvKLnCkjH6XoyW8")

VERIFY_TOKEN = "HARI"  #Verify the webhook
GOOGLE_API_KEY = "AIzaSyCEEcLoTsgjgw5V1VDjzslSVcsq9GmBxlc" 
TOKEN = "EAAN8oq5Ch3ABO6z6b35a7mV2Ber6nrICsLDVkS3EOH0qZCRFaLZAQdm8ZAfHQMn9OrB4JCfngQRg55tEuKqnA3xuOp41ccZAZBx1tVDrQlCIHJzf1rjKhnJrlZAplPOXZAifoZAm1zcrItKr6LkjLZBmZBDFExylJfqz6RZCTZAV1LvckJvToaBjyLYZATlzw8Uj6UXSA3Tu23z6NNCpofl2oP4Tu"

# Initialize chatbot components
def initialize_chatbot():
    # Load data for the chatbot
    data = """BetterBots Overview:

        BetterBots is a pioneering company specializing in AI-driven chatbot solutions aimed at revolutionizing customer interaction and service across various industries. With a commitment to delivering personalized, efficient, and intelligent chatbot services, BetterBots offers a suite of innovative products designed to cater to diverse business needs and individual requirements.
        Products Overview:
        InfoBot:
        Flagship product offering instant information retrieval, news updates, weather forecasts, and image sharing capabilities.
        Powered by ChatGPT, enabling advanced natural language processing and understanding.
        Integrated within WhatsApp for accessibility and ease of use.
        Targeted at both individual users seeking instant information and businesses looking for AI-driven customer service solutions.
        Tiqkets:
        Sophisticated system designed to manage customer queues efficiently.
        Leverages AI to enhance the customer waiting experience and streamline service operations.
        Primarily targeted at businesses with physical service locations, such as clinics, restaurants, and retail stores.
        Intractiv (formerly OrgBot):
        Customizable bot solution tailored for enterprise-level operations.
        Offers features such as customer query handling, task management, and team collaboration.
        Designed to improve efficiency and automate communication processes for large corporations and enterprises.

        Detailed Description:

        BetterBots, founded with the vision of transforming customer interactions through cutting-edge AI technology, has established itself as a leader in the chatbot solutions industry. With a focus on innovation, reliability, and customer satisfaction, BetterBots continues to push boundaries and redefine the way businesses engage with their customers.

        InfoBot stands as the cornerstone of BetterBots' product lineup, offering a versatile solution for instant information retrieval and customer support. Powered by ChatGPT, InfoBot harnesses the power of artificial intelligence to provide users with real-time updates, weather forecasts, news articles, and image sharing capabilities. Its seamless integration with WhatsApp makes it accessible to a wide audience, from individual users seeking quick answers to businesses looking to enhance their customer service offerings.

        Tiqkets addresses the challenges faced by businesses in managing customer queues effectively. Leveraging AI technology, Tiqkets optimizes the queuing experience, reducing wait times and improving customer satisfaction. Whether it's a busy clinic, bustling restaurant, or crowded retail store, Tiqkets ensures that customers are served promptly and efficiently, enhancing their overall experience.

        Intractiv (formerly OrgBot) emerges as the ultimate tool for enterprise-level operations, offering a customizable solution for large corporations to streamline communication and collaboration. With features such as customer query handling, task management, and team collaboration tools, Intractiv empowers organizations to improve efficiency, automate processes, and drive productivity across departments.

        In summary, BetterBots is committed to empowering businesses with intelligent chatbot solutions that enhance customer satisfaction, streamline operations, and drive growth. With a diverse range of products tailored to different industries and use cases, BetterBots continues to innovate and deliver value to its clients worldwide."""
    
    # Split text into chunks for processing
    text_chunks = split_text(data)
    
    # Download embeddings for semantic understanding
    embeddings = download_embeddings()
    
    # Add vectors to the vector store
    vector_store = add_vectors(text_chunks, embeddings)
    
    # Initialize the chatbot model
    model = initialize_chat_model()

    # Initialize QA chain for retrieving answers
    qa_chain = initialize_qa_chain(model, vector_store)
    
    return qa_chain

# Function to split text into manageable chunks
def split_text(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
    text_chunks = text_splitter.split_text(extracted_data)
    return text_chunks

# Function to download embeddings for semantic understanding
def download_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    return embeddings

# Function to add vectors to the vector store
def add_vectors(texts, embeddings):
    vector_store = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    print("Successfully added vectors")
    return vector_store

# Function to initialize the chatbot model
def initialize_chat_model():
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY,
                                 temperature=0.5, convert_system_message_to_human=True)
    return model

# Function to initialize the QA chain for retrieving answers
def initialize_qa_chain(model, vector_store):
    prompt_template = """
    Use the following context to answer the question at the end. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. Keep the answer concise.
    Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_store,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Function to convert text to Markdown format for better display
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Function to process incoming messages from WhatsApp
def process_message(message):
    try:
        user_query = message['text']
        qa_chain = initialize_qa_chain()
        result = qa_chain({"query": user_query})
        response_text = result["result"]
        formatted_response = to_markdown(response_text)
        return formatted_response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "An error occurred while processing your request. Please try again later."
    

URL = 'https://graph.facebook.com/v19.0/253420224531557/messages'
WABATOKEN = "EAAN8oq5Ch3ABO6z6b35a7mV2Ber6nrICsLDVkS3EOH0qZCRFaLZAQdm8ZAfHQMn9OrB4JCfngQRg55tEuKqnA3xuOp41ccZAZBx1tVDrQlCIHJzf1rjKhnJrlZAplPOXZAifoZAm1zcrItKr6LkjLZBmZBDFExylJfqz6RZCTZAV1LvckJvToaBjyLYZATlzw8Uj6UXSA3Tu23z6NNCpofl2oP4Tu"

import json
import logging

def send_msg(incoming_num, incoming_name, msg):
  url = URL
  wabaToken = WABATOKEN

  payload = json.dumps({
      "messaging_product": "whatsapp",
      "to": incoming_num,
      "type": "text",
      "text": {
          "preview_url": "false",
          "body": msg
      }
  })
  headers = {
      'Authorization': 'Bearer ' + wabaToken,
      'Content-Type': 'application/json'
  }

  try:
    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
  except requests.exceptions.RequestException as e:
    logging.error("Error sending message:", e)


@app.route('/webhook', methods=['GET'])
def verify():
    print("Webhook intilaized")
    mode = request.args.get('hub.mode')
    challenge = request.args.get('hub.challenge')
    token = request.args.get('hub.verify_token')

    send_msg(919567578391, 'hari', 'hello get')
    

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            return challenge, 200
        else:
            return "Verification failed", 403

@app.route('/webhook', methods=['POST'])
def webhook():
    print("POST is working")

    send_msg(919567578391, 'hari', 'hello post')
    
    
    body_param = request.json
    initialize_chatbot()

    if 'entry' in body_param:
        for entry in body_param['entry']:
            for change in entry.get('messaging', []):
                if 'message' in change:
                    phone_number_id = change['sender']['id']
                    from_number = change['sender']['phone_number']['formatted']
                    message_body = change['message']['text']
                    response = process_message({'text': message_body})

                    print("Phone number: ", phone_number_id)
                    print("from_number: ", from_number)
                    print("message_body: ", message_body)
                    return "Message processed successfully", 200
    

    return "Invalid request", 400



def send_whatsapp_message(phone_number_id, from_number, message):
    response = requests.post(
        f"https://graph.facebook.com/v13.0/253420224531557/messages?access_token={TOKEN}",
        json={
            "messaging_product": "whatsapp",
            "to": from_number,
            "text": {"body": message}
        },
        
        headers={"Content-Type": "application/json"}
    )
    print(message)

    if response.status_code == 200:
        print("Message sent successfully")
    else:
        print("Failed to send message")

public_url = ngrok.connect(5000).public_url
print(public_url)



if __name__ == '__main__':
    app.run(port=5000)

from flask import Flask, request, jsonify
from mistralai import Mistral
from langchain.memory import ConversationBufferMemory
from flask import Flask, request, jsonify
from flask_cors import CORS
# Your existing imports...
from mistralai import Mistral
import os

app = Flask(__name__)
CORS(app)

# API Key for Mistral client
api_key = 'buz8Beul0it0sCFDttDSgv63Ta3oHxxT'
client = Mistral(api_key=api_key)

# Global memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to interact with Mistral model using LangChain
def ask_mistral(question, chat_history):
    signed_url = get_signed_url()
    ocr_response = get_ocr_response(signed_url.url)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant who personally knows Kushal Bansal. "
                "You answer questions confidently and naturally based on your deep understanding of them. "
                "Respond as if you are a human, not a chatbot. Do not mention retrieving information or being an AI. Just answer naturally and conversationally. "
                f"Here is everything you know about kushal:{ocr_response}"
                "Now answer the user's questions about kushal bansal in a way that shows familiarity and certainty."
            ),
        }
    ] + chat_history + [{"role": "user", "content": question}]

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        max_tokens=512,
        messages=messages
    )

    return chat_response.choices[0].message.content


# API endpoint to get bot response
@app.route('/get_response', methods=['POST'])
def get_bot_response():
    # Get the question from the request JSON
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Get current chat history
    history = memory.load_memory_variables({})["chat_history"]
    chat_history = []
    for msg in history:
        if msg.type == "human":
            chat_history.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            chat_history.append({"role": "assistant", "content": msg.content})

    # Get response from Mistral model
    response = ask_mistral(question, chat_history)
    
    # Save the conversation to memory
    memory.save_context({"input": question}, {"output": response})

    return jsonify({'response': response})


# Helper function to retrieve OCR content
def get_ocr_response(signed_url):
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url,
        }
    )
    return ocr_response


# Helper function to get the signed URL of the document
def get_signed_url():
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "Kushal_Bansal.pdf",
            "content": open("Kushal_Bansal.pdf", "rb"),
        },
        purpose="ocr"
    )
    retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    return signed_url


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

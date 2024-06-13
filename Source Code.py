# Natural Language Query Agent
# This notebook implements a Natural Language Query Agent that can answer questions based on Stanford LLMs Lecture Notes and a table of LLM architectures.

# Import necessary libraries
!pip install openai
!pip install sentence-transformers
!pip install chromadb
import os
import csv
from typing import List, Dict
import openai
from sentence_transformers import SentenceTransformer
import chromadb
from google.colab import files
import io
import nltk
nltk.download('punkt')

# Install required packages
!pip install sentence-transformers chromadb python-dotenv

# Set up OpenAI API key
openai.api_key = "sk-proj-WV8QdfQVgqRue9UXxyLLT3BlbkFJUDoO1CPnE9gR7nQXIsQB"

# Data Loader
def load_lecture_notes(uploaded_files: List[io.IOBase]) -> List[Dict[str, str]]:
    notes = []
    for file in uploaded_files:
        content = file.getvalue().decode('utf-8')
        notes.append({"filename": file.name, "content": content})
    return notes

def load_model_architectures(file: io.IOBase) -> List[Dict[str, str]]:
    content = file.getvalue().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)

# Embedding Manager
class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("lecture_notes")

    def embed_and_index(self, documents: List[Dict[str, str]]):
        for doc in documents:
            sentences = nltk.sent_tokenize(doc['content'])
            for i, sentence in enumerate(sentences):
                embedding = self.model.encode(sentence)
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[sentence],
                    metadatas=[{"source": doc['filename'], "sentence_id": i}],
                    ids=[f"{doc['filename']}_{i}"]
                )

    def search(self, query: str, n_results: int = 3):
        query_embedding = self.model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results

# Query Processor
def process_query(query: str, conversation_history: List[Dict[str, str]]) -> str:
    prompt = f"Given the following conversation history and a new query, generate a structured query that can be used to retrieve relevant information:\n\n"
    
    for item in conversation_history:
        prompt += f"Human: {item['query']}\nAI: {item['response']}\n\n"
    
    prompt += f"Human: {query}\nStructured query:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

# Response Generator
def generate_response(query: str, relevant_info: List[str], conversation_history: List[Dict[str, str]]) -> str:
    prompt = "You are an AI assistant answering questions about LLMs based on lecture notes and a table of model architectures. "
    prompt += "Provide a conversational response to the query, using the relevant information provided. "
    prompt += "Include citations by mentioning the source of the information used in your response.\n\n"

    prompt += "Conversation history:\n"
    for item in conversation_history:
        prompt += f"Human: {item['query']}\nAI: {item['response']}\n\n"

    prompt += f"Human: {query}\n"
    prompt += "Relevant information:\n"
    for info in relevant_info:
        prompt += f"- {info}\n"
    prompt += "\nAI:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def generate_summary(conversation_history: List[Dict[str, str]]) -> str:
    prompt = "Generate a summary of the following conversation, including key points and potential flashcards or study tips:\n\n"
    
    for item in conversation_history:
        prompt += f"Human: {item['query']}\nAI: {item['response']}\n\n"
    
    prompt += "Summary:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

# Main function
def main():
    print("Please upload the lecture notes (txt files) and model architectures (csv file).")
    uploaded = files.upload()

    lecture_notes = []
    model_architectures = None

    for filename, file in uploaded.items():
        if filename.endswith('.txt'):
            lecture_notes.append({"filename": filename, "content": file.getvalue().decode('utf-8')})
        elif filename.endswith('.csv'):
            model_architectures = load_model_architectures(file)

    if not lecture_notes or not model_architectures:
        print("Error: Missing required files. Please upload lecture notes (txt) and model architectures (csv).")
        return

    # Initialize embedding manager and index documents
    embedding_manager = EmbeddingManager()
    embedding_manager.embed_and_index(lecture_notes)

    conversation_history = []

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        # Process query
        structured_query = process_query(query, conversation_history)

        # Retrieve relevant information
        search_results = embedding_manager.search(structured_query)
        relevant_info = [result['document'] for result in search_results['documents'][0]]

        # Generate response
        response = generate_response(query, relevant_info, conversation_history)
        print("AI:", response)

        # Update conversation history
        conversation_history.append({"query": query, "response": response})

        if len(conversation_history) % 5 == 0:
            summary = generate_summary(conversation_history)
            print("\nConversation Summary:")
            print(summary)
            print()

if __name__ == "__main__":
    main()

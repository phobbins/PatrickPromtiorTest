import os

# Set the USER_AGENT environment variable to specify a custom user agent for HTTP requests
os.environ['USER_AGENT'] = 'MyCustomUserAgent/1.0'

# Supress the warning from Hugging Face transformers library, specifically about the clean_up_tokenization_spaces parameter
import warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")


from typing import TypedDict
from operator import itemgetter


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


# URLs to scrape content from
URLs = [
    'https://www.promtior.ai/',
    'https://www.promtior.ai/service',
    'https://www.promtior.ai/use-cases',
    'https://www.promtior.ai/contacto'
]

# Initialize a loader to fetch documents (data) from the specified URLs
loader = WebBaseLoader(URLs)

# Load documents (data) from the URLs
docs = loader.load()

# Initialize a text splitter to divide data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter()

# Split the loaded data into manageable chunks
chunks = text_splitter.split_documents(docs)

# Initialize an embedding model from HuggingFace
embeddings = HuggingFaceEmbeddings()

# Create a FAISS vector store from the data chunks using the embeddings
vector = FAISS.from_documents(chunks, embeddings)

# Configure the vector store as a retriever to fetch relevant chunks based on a query
retriever = vector.as_retriever()

# Define a template for generating answers based on context and questions
template = """
Answer given the following context:
{context}

Question: {question}
"""

# Initialize a prompt template using the defined template
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Initialize a language model (LLM) using the Ollama API with the "llama2" model
llm = Ollama(model="llama2", temperature=0)

# Define the input schema for the retrieval-augmented generation (RAG) process
class RagInput(TypedDict):
    question: str

# Define the final processing chain for the RAG system
final_chain = (
    { 
        "context": itemgetter("question") | retriever,  # Retrieve context based on the question
        "question": itemgetter("question")              # Extract the question
    }
    | ANSWER_PROMPT    # Use the prompt template to format the input
    | llm              # Generate an answer using the language model
    | StrOutputParser() # Parse the output into a string
).with_types(input_type=RagInput)  # Define the expected input type







'''
# Funcionalidad de memoria de mensajes
# In-memory storage for session-based message history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or initialize the chat message history for a given session.

    Args:
        session_id (str): Unique identifier for the session.

    Returns:
        BaseChatMessageHistory: The chat history object for the session.
    """
    # If the session does not exist in the store, initialize it with a new ChatMessageHistory instance
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    # Return the chat message history for the session
    return store[session_id]

# Integrate the final processing chain with session-based message history functionality
final_chain_with_memory = RunnableWithMessageHistory(
    final_chain,
    get_session_history,
    input_messages_key="question",  # Key used to identify the input messages
    history_messages_key="context", # Key used to store and retrieve message history
).with_types(input_type=RagInput)  # Define the expected input type for the chain


'''

'''
#Multiquery
multiquery = MultiQueryRetriever.from_llm(retirever=retriever, llm=llm)

final_chain = (
    {
        "context": itemgetter("question") | multiquery, 
        "question": itemgetter("question")
    } 
    | ANSWER_PROMPT 
    | llm 
    | StrOutputParser()
).with_types(input_type = RagInput)

'''

'''
# Para correr aca directo
final_chain = {"context": itemgetter("question") | retriever, "question": itemgetter("question")} | ANSWER_PROMPT | llm | StrOutputParser()

FINAL_CHAIN_INVOKE = final_chain.invoke({"question": "Are you a chatbot with or without memory?"})


print(FINAL_CHAIN_INVOKE)



#Otra forma para correr con memoria
chat_history = []

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer given the following context:
            {context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

llm = Ollama(model="llama2", temperature=0)


chain = ANSWER_PROMPT | llm

final_chain = (
    {
        "context": itemgetter("question") | retriever, 
        "question": itemgetter("question")
    } 
    | ANSWER_PROMPT 
    | llm 
    | StrOutputParser()
).with_types(input_type = RagInput)
'''
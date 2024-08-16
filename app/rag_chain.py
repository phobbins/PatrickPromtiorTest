from typing import TypedDict
from operator import itemgetter
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'MyCustomUserAgent/1.0'


URLs=[
    'https://www.promtior.ai/',
    'https://www.promtior.ai/service',
    'https://www.promtior.ai/use-cases',
    'https://www.promtior.ai/contacto'

]

loader = WebBaseLoader(URLs)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()

chunks = text_splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings()

vector = FAISS.from_documents(chunks, embeddings)

retriever = vector.as_retriever()

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = Ollama(model="llama2", temperature=0)

class RagInput(TypedDict):
    question: str



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
final_chain = {"context": itemgetter("question") | retriever, "question": itemgetter("question")} | ANSWER_PROMPT | llm | StrOutputParser()

FINAL_CHAIN_INVOKE = final_chain.invoke({"question": "When was Promtior created?"})


print(FINAL_CHAIN_INVOKE)
'''
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 20:21:30 2024

@author: subhr
"""
#!pip install chromadb

import langchain
import os
import openai

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import nltk
from PyPDF2 import PdfReader

nltk.download("punkt")
os.environ["OPENAI_API_KEY"] = "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"
loader = UnstructuredFileLoader('Downloads/LopaResume.pdf')
#loader = UnstructuredFileLoader('Downloads/sample-layout.pdf', mode='elements')
documents= loader.load()
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'])
doc_search = Chroma.from_documents(texts,embeddings)
chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)
query = "what is lopa's previous experience?"
chain.run(query)




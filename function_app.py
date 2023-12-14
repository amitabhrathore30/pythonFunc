import azure.functions as func
import logging
import openai
import requests
import jsons
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import (RetrievalQA, RetrievalQAWithSourcesChain,
                              ConversationalRetrievalChain, LLMChain, ChatVectorDBChain)
from langchain.memory import ConversationBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from typing import Any, List
from pydantic import Field
import pytesseract
from PIL import Image
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load.serializable import Serializable
import copy
from langchain.embeddings import AzureOpenAIEmbeddings
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="getDataFromPineconeDocs")
def getDataFromPineconeDocs(req: func.HttpRequest) -> func.HttpResponse:
    system_prompt = '''
As a data scientist specializing in clinical terminology, you are tasked with leveraging your expertise to extract and filter relevant data from a vector database. The database contains a wealth of information related to clinical terms, and your goal is to perform specific tasks to facilitate further analysis. 
You are proficient in analyzing different criterias given to you and filter the data accordingly.
Remember to prioritize efficiency, clarity, and the ability to derive valuable insights from the clinical terminology data in the vector database.
'''
    query = req.params.get('query')
    os.environ["OPENAI_API_BASE"] = "https://coaa-openai-service.openai.azure.com/"
    os.environ["OPENAI_API_TYPE"] = "azure" #"azure_ad"
    os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
    os.environ["OPENAI_API_KEY"] = "e3f6623c42b349b888ec8fe6315eff2c"
    openai.api_type = os.environ["OPENAI_API_TYPE"]
    openai.base_url = os.environ["OPENAI_API_BASE"]
    openai.api_version = os.environ["OPENAI_API_VERSION"]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = AzureChatOpenAI(
            openai_api_base="https://coaa-openai-service.openai.azure.com/",
            openai_api_version="2023-07-01-preview",
            deployment_name="gpt-35-turbo-16k",
            openai_api_key="e3f6623c42b349b888ec8fe6315eff2c",
            openai_api_type="azure"
        )
    embedding_deployment_name = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(model=embedding_deployment_name)
    pinecone.init(
           api_key="bb9d0a52-f399-4223-8ac0-19c18af56da5",
           environment="us-west1-gcp-free")

    pinecone_index_name = "poc-documents-index"
    index = Pinecone.from_existing_index(pinecone_index_name, embeddings, namespace="ns_scrapedDataJson2")
    finalJson=[]
    for x in range(10):
        answer =  RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=index.as_retriever(search_kwargs={'filter': {'id': str(x+1)}})
            )
        ans = str(answer(query))
        finalJson.append(ans)

    return func.HttpResponse(
        json.dumps(finalJson), status_code=200
    )

def SearchViaRetrievalQA(index, query, llm):
    answer = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever()
    )
    ans = answer(query)
    return ans["result"]


def SearchViaRetrievalQAWithSourcesChain(index, query, llm):
    answer = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever()
    )
    ans = str(answer(query))

    return ans


def SearchViaConversationalRetrievalChain(index, query, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm, index.as_retriever(), memory=memory)
    result = qa({"question": query})
    ans = result["answer"]
    return ans


def SearchViaConversationalRetrievalChainWithSources(index, query, llm):
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    chain = ConversationalRetrievalChain(
        retriever=index.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    ans = result["answer"]
    return ans


def SearchViaQAChain(index, query, score, llm, k=50):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)

    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer


def SearchViaVectorDBChain(index, query, llm, k=1):
    pdfQA = ChatVectorDBChain.from_llm(
        llm, index, return_source_documents=True)
    result = pdfQA({"question": query, "chat_history": ""})
    return result["answer"]
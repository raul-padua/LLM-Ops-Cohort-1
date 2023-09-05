import os
import arxiv
import httpx  # New import
import chainlit as cl
from chainlit import user_session
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# New Constants for FastAPI Service
API_HOST = "web"  # Service name from docker-compose.yaml
API_PORT = 80  # Port specified in docker-compose.yaml

# New function to generate text using FastAPI service
async def async_generate_text(prompt):
    async with httpx.AsyncClient() as client:
        data = {"prompt": prompt}
        response = await client.post(f"http://{API_HOST}:{API_PORT}/generateText/", json=data)
        result = response.json()
        return result["task_id"]

# New function to get task status from FastAPI service
async def async_get_task_status(task_id):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://{API_HOST}:{API_PORT}/generateTextTask/{task_id}")
        return response.text

@cl.langchain_factory(use_async=True)
async def init():
    arxiv_query = None
    while arxiv_query == None:
        arxiv_query = await cl.AskUserMessage(
            content="Please enter a topic to begin!", timeout=15
        ).send()

    search = arxiv.Search(
        query=arxiv_query["content"],
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    await cl.Message(content="Downloading and chunking articles...").send()
    pdf_data = []
    for result in search.results():
        loader = PyMuPDFLoader(result.pdf_url)
        loaded_pdf = loader.load()
        for document in loaded_pdf:
            document.metadata["source"] = result.entry_id
            document.metadata["file_path"] = result.pdf_url
            document.metadata["title"] = result.title
            pdf_data.append(document)

    embeddings = OpenAIEmbeddings(disallowed_special=(),)
    docsearch = Chroma.from_documents(pdf_data, embeddings)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0,
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    await cl.Message(
        content=f"We found a few papers about `{arxiv_query['content']}` you can now ask questions!"
    ).send()

    # New code for generating text and retrieving status
    await cl.Message(content="Waiting for text generation...").send()
    prompt = await cl.AskUserMessage(
        content="Please enter a text generation prompt!", timeout=15
    ).send()
    task_id = await async_generate_text(prompt["content"])

    while True:
        status = await async_get_task_status(task_id)
        if "Task Pending" not in status:
            await cl.Message(content=f"Text generation result: {status}").send()
            break

    return chain

@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]
    source_elements_dict = {}
    source_elements = []
    for idx, source in enumerate(res["source_documents"]):
        title = source.metadata["title"]
        if title not in source_elements_dict:
            source_elements_dict[title] = {
                "page_number": [source.metadata["page"]],
                "url": source.metadata["file_path"],
            }
        else:
            source_elements_dict[title]["page_number"].append(source.metadata["page"])
        source_elements_dict[title]["page_number"].sort()

    for title, source in source_elements_dict.items():
        page_numbers = ", ".join([str(x) for x in source["page_number"]])
        text_for_source = f"Page Number(s): {page_numbers}\nURL: {source['url']}"
        source_elements.append(
            cl.Text(name=title, content=text_for_source, display="inline")
        )
    await cl.Message(content=answer, elements=source_elements).send()

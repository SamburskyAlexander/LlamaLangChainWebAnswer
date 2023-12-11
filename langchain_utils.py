from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter


def get_text_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
        is_separator_regex=True,
        separators=['\d+(\.\d+){2}\.\s', '\d+\.\s', '\d+\.\d+\.\s', '\n', '\n\n'],
    )

def get_llm(model_path, n_ctx):
    return LlamaCpp(
        model_path=model_path,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        max_tokens=n_ctx,
        n_ctx=n_ctx,
        temperature=0.0,
        top_p=1,
        verbose=True
    )

def load_doc(doc_path, doc_type, loader, splitter):
    return DirectoryLoader(
        path=doc_path, glob=f"**/*.{doc_type}", loader_cls=loader
    ).load_and_split(text_splitter=splitter)

def load_all_docs(docs_dir_path, splitter):
    csv_doc_list = load_doc(
        docs_dir_path,
        "csv",
        CSVLoader,
        splitter
    )
    pdf_doc_list = load_doc(
        docs_dir_path,
        "pdf",
        PyPDFLoader,
        splitter
    )
    return csv_doc_list + pdf_doc_list

def get_db(model_name, docs):
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    return Chroma.from_documents(docs, embedder)

def get_retriever(db_with_vectors, search_kwargs):
    return = db_with_vectors.as_retriever(search_kwargs=search_kwargs)

def get_langchain(chat_prompt_string, model, retriever):
    return {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    } | ChatPromptTemplate.from_template(chat_prompt_string) | model | StrOutputParser()

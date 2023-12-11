from langchain_utils import get_text_splitter
from langchain_utils import get_llm
from langchain_utils import load_all_docs
from langchain_utils import get_db
from langchain_utils import get_retriever
from langchain_utils import get_langchain

from pydantic import BaseModel
from fastapi import FastAPI
    
# ===============================
# CONSTANTS FOR LANHCHAIN PROCESS
# ===============================

DOCS_DIR = "docs/"
MODEL_DIR = "models/"
LMM_PATH_NAME = MODEL_DIR + "llama-2-7b-chat.Q4_K_M.gguf"
EMB_PATH_NAME = "ai-forever/ruElectra-large"
SPLIT_CHUNK_SIZE = 500
SPLIT_CHUNK_OVERLAP = 0
LLM_N_CTX = 3500
LLM_RETRIEVER_K = 3

CHAT_PROMPT = """<<SYS>>Ты бот-ассистент, отвечающий на вопросы по услугам и продуктам банка в соответствии с утверрждёнными документами.<</SYS>>
[INST]Вопрос: {question}[/INST]
<<SYS>>Подходящие документы: {context}<</SYS>>
Ответ:
"""

# ===============================
# MAIN
# ===============================

class MessageEntry(BaseModel):
    msg: str
    uid: str

def run_main():
    llm_text_splitter = get_text_splitter(
        chunk_size=SPLIT_CHUNK_SIZE,
        chunk_overlap=SPLIT_CHUNK_OVERLAP
    )
    llm = get_llm(
        model_path=LMM_PATH_NAME,
        n_ctx=LLM_N_CTX
    )
    loaded_docs = load_all_docs(
        docs_dir_path=DOCS_DIR,
        splitter=llm_text_splitter
    )
    db_with_vectors = get_db(
        model_name=EMB_PATH_NAME,
        docs=loaded_docs
    )
    llm_retriever = get_retriever(
        db_with_vectors=db_with_vectors,
        search_kwargs={"k": LLM_RETRIEVER_K}
    )
    llm_langchain = get_langchain(
        chat_prompt_string=CHAT_PROMPT,
        model=llm,
        retriever=llm_retriever
    )

    app = FastAPI()
    @app.post("/message")
    async def handle_message(data: MessageEntry):
        return {"response": f"{llm_langchain.invoke({'question': data.msg})}"}


run_main()

## Code Source -- https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/
## Code Source -- https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/
## Code Source -- https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
## Code Source -- https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/

import streamlit as st
import os

from sqlalchemy import create_engine
import pandas as pd

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms import Ollama as cls_ollama_langchain
#from document_loader import ChromaCRUD 
# from ollama_langchain.ollama_ingest_models import get_list_of_models
# from ollama_langchain.ollama_llm import OLLAMA_LLM
import ollama as ollama_py_lib

from utils.ollama_ingest_models import get_list_of_models
from utils.ollama_llm import OLLAMA_LLM
from utils.document_loader import ChromaCRUD 
from nl2sql_langchain.invoke_chain import Nl2SQL_Langchain
from nl2sql_llama_idx.invoke_nl2sql_llamaIdx import Nl2SQL_LLamaIndex
#from hf_models.hf_trfmr_models import HFModelsInvoke
## TODO -- ImportError: Loading a BitNet quantized model requires accelerate (`pip install accelerate`)



from utils.data_ingest import Nl2SQL_DataIngest
from utils.util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

EMBEDDING_MODEL = "nomic-embed-text"
PATH = "pdf_dir"
logger.debug("--EMBEDDING_MODEL->> %s",type(EMBEDDING_MODEL))

st.title("Langchain_n_LlamaIndex_exp...")
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()
    logger.debug("-init--st.session_state--->> %s",st.session_state)


selected_model_1 = st.sidebar.selectbox("Model to Download:", st.session_state["list_of_models"])

selected_model = st.sidebar.selectbox("Select a model:", st.session_state["list_of_models"])
if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = cls_ollama_langchain(model=selected_model)

logger.debug("-init-a-st.session_state--->> %s",st.session_state)

folder_path = st.sidebar.text_input("Provide PDF DIR path:", PATH) # st.warning("ERROR -- HELLO---AA")
if folder_path:
    logger.debug("-local Docs Loading from Folder Path ->> %s",folder_path) #st.warning("ERROR -- HELLO---BB")
    if not os.path.isdir(folder_path):
        st.error("No path exists - Provide Data PDF Files path.")
    else:
        if st.sidebar.button("Index Documents"):
            if "db" not in st.session_state:
                with st.spinner("Started - load_documents_into_database -- Embedding n loading into Chroma DB."):
                    st.session_state["db"] = ChromaCRUD().load_documents_into_database(
                                                EMBEDDING_MODEL, 
                                                folder_path
                                            )
                st.info("START - QnA with Local RAG Docs...")
else:
    st.warning("ERROR -- Provide Data PDF Files path.")
    logger.error("-NO LOcal -Data PDF Files path ->> %s",folder_path)

# Init the Chat_History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

model_name_1 = "llama3.2"
embedding_model_name_1 = "nomic-embed-text"
model_name_0 = "llama3.1"
dataset_name = "mtcars"
list_of_datasets = ["mtcars","tips","UKgas","airquality"]
st.session_state["list_of_datasets"] = list_of_datasets


if "sqlite_db" not in st.session_state:
    dataset_name = st.sidebar.selectbox("Select Dataset Name :", st.session_state["list_of_datasets"])
    st.session_state["dataset_name"] = dataset_name
    langc_sql_db_name , sql_alchemy_engine , sqlite_tb_name = Nl2SQL_DataIngest().get_alchemy_engine(dataset_name)
    st.session_state["sqlite_db"] = langc_sql_db_name
    st.session_state["sql_alchemy_engine"] = sql_alchemy_engine
    st.session_state["sqlite_tb_name"] = sqlite_tb_name

    logger.debug("-init--st.session_state--sqlite_db->> %s",st.session_state)

if st.chat_input != None:
    logger.debug("-STARTED-chat_input->>")

    if user_nl_query := st.chat_input("Question"):
        st.session_state.messages.append({"role": "user", "content": user_nl_query})
        with st.chat_message("user"):
            st.markdown(user_nl_query)

        # nl2sql_response = Nl2SQL_Langchain().invoke_nl2sql(user_nl_query,st.session_state["sqlite_db"])
        # logger.debug("--Nl2SQL_Langchain--nl2sql_response->> %s",type(nl2sql_response))
        #st.info(nl2sql_response)
        #Nl2SQL_Langchain().invoke_sql_db_toolkit(user_nl_query,st.session_state["sqlite_db"])
        llm_llama_idx , embed_model_llama_idx  = Nl2SQL_DataIngest().invoke_ollama_llama_idx(embedding_model_name_1,
                                                                                                        model_name_1,True)
        #st.info(nl2sql_response)
        sql_db_llama_idx = Nl2SQL_DataIngest().get_llama_idx_sqldb(embedding_model_name_1,
                                                alchemy_engine= st.session_state["sql_alchemy_engine"],
                                                table_name=st.session_state["sqlite_tb_name"])
        logger.debug("---get_llama_idx_sqldb--TYPE-sql_db_llama_idx--aa->> %s",type(sql_db_llama_idx))
        response_sql_retr_eng , dict_res_sql_retr_eng = Nl2SQL_LLamaIndex().wrapper_get_query(user_nl_query,
                                                                            embedding_model_name_1,
                                                                            sql_db_llama_idx,
                                                                            st.session_state["sqlite_tb_name"]
                                                                            )

        st.info("Initial RES - from -- SQLTableRetrieverQueryEngine")
        st.info(response_sql_retr_eng)
        st.info(str(dict_res_sql_retr_eng))

        
        #Nl2SQL_LLamaIndex().parse_error(
        
        # nl2sql_stream = OLLAMA_LLM.get_streaming_chain(st.session_state["llm_ollama_langchain"],
        #                                     st.session_state["sqlite_db"],
        #                                     user_nl_query,
        #                                     #st.session_state.messages,
        #                                     )
        # logger.debug("-get_streaming_chain-STREAM->> %s",nl2sql_stream) ##<generator object RunnableSequence.stream at 0x7f078e30f100>
        # response = st.write_stream(nl2sql_stream)
        # logger.debug("-get_streaming_chain-response->> %s",type(response))
        # st.session_state.messages.append({"role": "assistant", "content": response})

from utils.util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

import streamlit as st

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms import Ollama as cls_ollama_langchain
#from document_loader import ChromaCRUD 
# from ollama_langchain.ollama_ingest_models import get_list_of_models
# from ollama_langchain.ollama_llm import OLLAMA_LLM
import ollama as ollama_py_lib

mtcars_db = SQLDatabase.from_uri("sqlite:///mtcars.db")
model_name_1 = "llama3.2"
model_name_0 = "llama3.1"


class Nl2SQL_Langchain:
    """
    """

    def invoke_nl2sql(self,user_nl_query,sqlite_local_db):
        """
        """
        #TODO - if Missed Creating Embeddings and DB Chroma  -- if "db" in st.session_state:
        llm_ollama_langchain = cls_ollama_langchain(model=model_name_1)
        st.session_state["llm_ollama_langchain"] = llm_ollama_langchain

        chain = create_sql_query_chain(llm_ollama_langchain, sqlite_local_db)
        nl2sql_response = chain.invoke({"question": user_nl_query})
        logger.debug(f'-TYPE--response-->> {type(nl2sql_response)}')
        logger.debug(f'-response-->> {nl2sql_response}')
        return nl2sql_response

    def invoke_sql_db_toolkit(self,user_nl_query,sqlite_local_db):
        """
        """
        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        llm_ollama_langchain = cls_ollama_langchain(model=model_name_1)
        #st.session_state["llm_ollama_langchain"] = llm_ollama_langchain
        toolkit = SQLDatabaseToolkit(db=sqlite_local_db, llm=llm_ollama_langchain)
        tools = toolkit.get_tools()
        logger.debug(f'---SQLDatabaseToolkit--tools-->> {tools}')
        
        


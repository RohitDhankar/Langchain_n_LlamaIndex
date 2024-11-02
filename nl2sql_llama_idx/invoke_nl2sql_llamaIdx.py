
# Code SOURCE - https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/
# Code Source - https://docs.llamaindex.ai/en/stable/examples/llm/ollama/


import streamlit as st
import os

from utils.util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatResponse

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.bridge.pydantic import BaseModel, Field
#from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama as ollama_llama_index
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import Settings as llama_idx_settings
from llama_index.core.query_engine import NLSQLTableQueryEngine

from llama_index.core.retrievers import SQLRetriever
from typing import List
from llama_index.core.query_pipeline import FnComponent

from llama_index.core import VectorStoreIndex , SQLDatabase
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
                            SQLTableNodeMapping,
                            ObjectIndex,
                            SQLTableSchema,
                        )

from llama_index.core import VectorStoreIndex
from utils.data_ingest import Nl2SQL_DataIngest


from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

from .llama_idx_prompt_tempates import (AUTO_TABLE_DESC)


## TODO === 
# from .nl2sql_llamaidx_utils import (get_table_context_str,
#                                     parse_response_to_sql)


model_name_1 = "llama3.2"
model_name_0 = "llama3.1"

class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(..., description="table name (must be underscores and NO spaces)")
    table_summary: str = Field(..., description="short, concise summary/caption of the table")


class Nl2SQL_LLamaIndex:
    """
    Desc:
        - Nl2SQL_LLamaIndex
    """
    def __new__(cls):
        if not hasattr(cls,'instance'):
            cls.instance = super(Nl2SQL_LLamaIndex,cls).__new__(cls)

        return cls.instance

    @classmethod
    def wrapper_get_query(self,
                        user_nl_query , 
                        embedding_model_name_1,
                        sql_db_llama_idx,
                        table_name
                        ):
        """
        """
        try:

            ls_table_schema_objs = []
            ls_table_names = [table_name]
            llm_llama_idx , embed_model_llama_idx = Nl2SQL_DataIngest().invoke_ollama_llama_idx(embedding_model_name_1,
                                                                        model_name_1,
                                                                        json_mode=False)
            llama_idx_settings.llm = llm_llama_idx
            llama_idx_settings.embed_model = embed_model_llama_idx

            tables_nodes_mapped = SQLTableNodeMapping(sql_db_llama_idx)
            obj_sql_tab_scehma = SQLTableSchema(table_name=table_name) 

            for iter_tab_name in ls_table_names:
                ls_table_schema_objs.append(SQLTableSchema(table_name=iter_tab_name))
            logger.debug("--ls_table_schema_objs--->> %s",ls_table_schema_objs)
            #[SQLTableSchema(table_name='mtcars_name_df', context_str=None)]
            obj_index = ObjectIndex.from_objects(
                                    ls_table_schema_objs , #table_schema_objs
                                    tables_nodes_mapped,
                                    VectorStoreIndex,
                                )
            logger.debug("--obj_index--->> %s",obj_index) #<llama_index.core.objects.base.ObjectIndex object at 0x7f7797e6fe90>
            sql_tab_retr_query_engine = SQLTableRetrieverQueryEngine(sql_db_llama_idx, 
                                        obj_index.as_retriever(similarity_top_k=1)
                                        )
            logger.debug("--sql_tab_retr_query_engine--->> %s",sql_tab_retr_query_engine) # OK - <llama_index.core.indices.struct_store.sql_query.SQLTableRetrieverQueryEngine object at 0x7fdf81cbbec0>
            response_sql_retr_eng = sql_tab_retr_query_engine.query(user_nl_query)
            #logger.debug("-TYPE==-response_sql_retr_eng--->> %s",type(response_sql_retr_eng)) # OK - <class 'llama_index.core.base.response.schema.Response'>
            logger.debug("--response_sql_retr_eng--->> %s",response_sql_retr_eng) # OK - "According to our database, there are 32 records related to cars in our dataset."
            #.get_formatted_sources
            # you can also fetch the raw result from SQLAlchemy!
            if response_sql_retr_eng.metadata["result"]:
                metadata_result = response_sql_retr_eng.metadata["result"] # NO errors Out 
                logger.debug("--metadata_res--->> %s",metadata_result)

            res_source_nodes = response_sql_retr_eng.source_nodes
            metadata_all_res = response_sql_retr_eng.metadata
            res_sql_retr_eng = response_sql_retr_eng.response
            logger.debug("--res_sql_retr_eng--->> %s",res_sql_retr_eng)
            
            logger.debug("--res_source_nodes--->> %s",res_source_nodes)
            logger.debug("--metadata_all_res--->> %s",metadata_all_res)

            sql_retriever = SQLRetriever(sql_db_llama_idx)
            logger.debug("--TYPE--sql_retriever--->> %s",type(sql_retriever))
            logger.debug("--sql_retriever--->> %s",sql_retriever)
            # Code SOURCE - https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/

            ## TODO -- 
            # table_parser_component = FnComponent(fn=get_table_context_str)
            # logger.debug("--TYPE--table_parser_component--->> %s",type(table_parser_component))
            # sql_parser_component = FnComponent(fn=parse_response_to_sql)
            # logger.debug("--TYPE--sql_parser_component--->> %s",type(sql_parser_component))
            #


            if "dataset_name" not in st.session_state:
                dataset_name = st.sidebar.selectbox("Select Dataset Name :", st.session_state["list_of_datasets"])
            else:
                dataset_name = st.session_state["dataset_name"]

            langc_sql_db_name , sql_alchemy_engine , sqlite_tb_name = Nl2SQL_DataIngest().get_alchemy_engine(dataset_name)
            text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
                                                        dialect=sql_alchemy_engine.dialect.name
                                                    )
            
            logger.debug("--text2sql_prompt.template--->> %s",text2sql_prompt.template)




            #NLSQLTableQueryEngine()


            #response_sql_retr_eng.get_formatted
            #response_sql_retr_eng.get_formatted_sources(500)

            dict_res_sql_retr_eng = {}
            
            return response_sql_retr_eng 
        except Exception as err:
            logger.error(f"--Error--wrapper_get_query->> {err}")
        

    # TODO = Done - Moved below to - ollama_langchain/utils/data_ingest.py
    # @classmethod
    # def invoke_ollama_llama_idx(self,
    #                             user_nl_query,
    #                             embedding_model_name_1,
    #                             model_name=model_name_1,
    #                             json_mode=True):
    #     """
    #     Desc:
    #         - 
    #     """

    #     llm_llama_idx = ollama_llama_index(model=model_name, 
    #                                         request_timeout=120.0,
    #                                         json_mode=json_mode)

    #     nl2sql_response = llm_llama_idx.complete(
    #                                         str(user_nl_query)
    #                                     )
    #     logger.debug("-init--invoke_ollama_llama_idx--TEST---response>> %s",nl2sql_response)
        
    #     embed_model_llama_idx = OllamaEmbedding(
    #                     model_name=embedding_model_name_1,
    #                     base_url="http://localhost:11434",
    #                     ollama_additional_kwargs={"mirostat": 0},
    #                 )

    #     # pass_embedding = ollama_embedding.get_text_embedding_batch(
    #     #     ["This is a passage!", "This is another passage"], show_progress=True
    #     # )
    #     # print(pass_embedding)

    #     # query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
    #     # print(query_embedding)

    #     return nl2sql_response, llm_llama_idx , embed_model_llama_idx 

    # TODO = Done - Moved below to - ollama_langchain/utils/data_ingest.py
    # @classmethod
    # def get_llama_idx_sqldb(self,
    #                         embedding_model_name_1,
    #                         alchemy_engine,
    #                         table_name):
    #     """
    #     """
    #     _ ,llm_llama_idx , embed_model_llama_idx = self.invoke_ollama_llama_idx(embedding_model_name_1,model_name_1,json_mode=False)
    #     llama_idx_settings.llm = llm_llama_idx
    #     llama_idx_settings.embed_model = embed_model_llama_idx
    #     sql_db_llama_idx = SQLDatabase(alchemy_engine,include_tables=[table_name])
    #     logger.debug("---get_llama_idx_sqldb--TYPE-sql_db_llama_idx--->> %s",type(sql_db_llama_idx))
    #     #<class 'llama_index.core.utilities.sql_wrapper.SQLDatabase'>
    #     return sql_db_llama_idx

    # def invoke_nl2sql_chain():
    #     """
    #     """
    #     llama_idx_settings.llm = 
    #     llama_idx_settings.embed_model = 


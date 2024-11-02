
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

import streamlit as st

from llama_index.core.retrievers import SQLRetriever
from typing import List
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.objects import (
                                SQLTableNodeMapping,
                                ObjectIndex,
                                SQLTableSchema,
                            )
from llama_index.core import SQLDatabase, VectorStoreIndex

#sql_retriever = SQLRetriever(sql_database)

from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatResponse

embedding_model_name_1 = "nomic-embed-text"
alchemy_engine= st.session_state["sql_alchemy_engine"]
table_name=st.session_state["sqlite_tb_name"]

#from utils.data_ingest import Nl2SQL_DataIngest
# sql_database = Nl2SQL_DataIngest().get_llama_idx_sqldb(embedding_model_name_1,
#                                                     alchemy_engine,
#                                                     table_name)

    
def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """
    """
    try:
        # context_strs = []
        # for table_schema_obj in table_schema_objs:
        #     table_info = sql_database.get_single_table_info(
        #         table_schema_obj.table_name
        #     )
        #     if table_schema_obj.context_str:
        #         table_opt_context = " The table description is: "
        #         table_opt_context += table_schema_obj.context_str
        #         table_info += table_opt_context

        #     context_strs.append(table_info)
        #     return_val = "\n\n".join(context_strs)
        return_val = "TODO---"

        return return_val
    except Exception as err:
            logger.error(f"--Error--wrapper_get_query->> {err}")


def parse_response_to_sql(response: ChatResponse):
    """
    Desc:
        - Parse response to SQL
    """
    try:
        response = response.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
            res_parsed_sql = response.strip().strip("```").strip()
        return res_parsed_sql
    except Exception as err:
        logger.error(f"--Error--parse_response_to_sql->> {err}")




# class Nl2SqlLlamaIdxUtils:
#     """
#     Desc:
#         - Nl2SqlLlamaIdxUtils
#     """
#     def __new__(cls):
#         if not hasattr(cls,'instance'):
#             cls.instance = super(Nl2SqlLlamaIdxUtils,cls).__new__(cls)

#         return cls.instance        
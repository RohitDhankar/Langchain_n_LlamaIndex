
from utils.util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from sqlalchemy import create_engine
import pandas as pd
from langchain_community.utilities import SQLDatabase as langc_sql_db
from langchain.chains import create_sql_query_chain
from langchain_community.llms import Ollama
from llama_index.core import Settings as llama_idx_settings
from llama_index.core import VectorStoreIndex , SQLDatabase


#from document_loader import ChromaCRUD 
# from ollama_langchain.ollama_ingest_models import get_list_of_models
# from ollama_langchain.ollama_llm import OLLAMA_LLM
import ollama as ollama_py_lib

#/ollama_langchain/nl2sql_llama_idx/invoke_nl2sql_llamaIdx.py
#from ollama_langchain.nl2sql_llama_idx.invoke_nl2sql_llamaIdx import Nl2SQL_LLamaIndex

model_name_1 = "llama3.2"
model_name_0 = "llama3.1"

class Nl2SQL_DataIngest:
    """
    """

    def get_alchemy_engine(self,dataset_name):
        """
        Desc:
            - returns - SQLDatabase
            - how many cars - whats table name 
            - for Merc 240D whats the mpg value
        """
        
        sqlite_csv_path = "././sqlite_data/"+str(dataset_name)+".csv" #ollama_langchain/sqlite_data/tips.csv
        df_dataset_name= pd.read_csv(sqlite_csv_path)
        logger.debug(df_dataset_name.info(verbose=True))
        alchemy_str = 'sqlite:///'+str(dataset_name)+'.db'
        sql_alchemy_engine = create_engine(alchemy_str, echo=False) #/// persist #// in memory
        sqlite_tb_name =  str(dataset_name) + "_name_df"
        # TODO - OperationalError: (sqlite3.OperationalError) object name reserved for internal use:
        #(Background on this error at: https://sqlalche.me/e/20/e3q8)

        df_dataset_name.to_sql(name=sqlite_tb_name, 
                                con=sql_alchemy_engine , 
                                index=False , 
                                if_exists="replace") 
                                
        langc_sql_db_name = langc_sql_db.from_uri(alchemy_str)
        logger.debug(f'--langc_sql_db_name.dialect--->> {langc_sql_db_name.dialect}') #--mtcars_db.dialect--->> sqlite
        # print(mtcars_db.get_usable_table_names())
        logger.debug(f'--langc_sql_db_name.get_usable_table_names()--->> {langc_sql_db_name.get_usable_table_names()}')
        # langc_sql_db_name.run("SELECT * FROM Artist LIMIT 10;")
        return langc_sql_db_name , sql_alchemy_engine , sqlite_tb_name


    @classmethod
    def get_llama_idx_sqldb(self,
                            embedding_model_name_1,
                            alchemy_engine,
                            table_name):
        """
        Desc:
            - moved method - REFACTOR 
            - 
        """
        try:
            model_name_1 = "llama3.2"
            llm_llama_idx , embed_model_llama_idx = self.invoke_ollama_llama_idx(embedding_model_name_1,
                                                                                model_name_1,json_mode=False)
            llama_idx_settings.llm = llm_llama_idx
            llama_idx_settings.embed_model = embed_model_llama_idx
            sql_db_llama_idx = SQLDatabase(alchemy_engine,include_tables=[table_name])
            logger.debug("---get_llama_idx_sqldb--TYPE-sql_db_llama_idx--->> %s",type(sql_db_llama_idx))
            #<class 'llama_index.core.utilities.sql_wrapper.SQLDatabase'>
            return sql_db_llama_idx
        
        except Exception as err:
            logger.error(f"--Error--get_llama_idx_sqldb->> {err}")

        #Nl2SQL_DataIngest
        # sql_database = Nl2SQL_LLamaIndex().get_llama_idx_sqldb(embedding_model_name_1,
        #                     alchemy_engine,
        #                     table_name)



    @classmethod
    def invoke_ollama_llama_idx(self,
                                embedding_model_name_1,
                                model_name=model_name_1,
                                json_mode=True):
        """
        Desc:
            - 
        """


        from llama_index.llms.ollama import Ollama as ollama_llama_index
        from llama_index.embeddings.ollama import OllamaEmbedding

        llm_llama_idx = ollama_llama_index(model=model_name, 
                                            request_timeout=120.0,
                                            json_mode=json_mode)

        # TODO -- just a test of the prompt - trigger -- not required 
        # nl2sql_response = llm_llama_idx.complete(str(user_nl_query))
        # logger.debug("-init--invoke_ollama_llama_idx--TEST---response>> %s",nl2sql_response)
        
        embed_model_llama_idx = OllamaEmbedding(
                        model_name=embedding_model_name_1,
                        base_url="http://localhost:11434",
                        ollama_additional_kwargs={"mirostat": 0},
                    )

        # pass_embedding = ollama_embedding.get_text_embedding_batch(
        #     ["This is a passage!", "This is another passage"], show_progress=True
        # )
        # print(pass_embedding)

        # query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
        # print(query_embedding)

        return llm_llama_idx , embed_model_llama_idx 



"""
# from ollama_langchain.ollama_ingest_models import get_list_of_models
# st.session_state["list_of_models"] = get_list_of_models()
#st.session_state["llm"] = Ollama(model=selected_model)

# model_name_1 = "llama3.2"
# model_name_0 = "llama3.1"
# logger.debug(f'-ollama_py_lib.list()--->> {ollama_py_lib.list()}')
# #logger.debug(f'--ollama_py_lib.show(model_name)--->> {ollama_py_lib.show(model_name)}')


# llm = Ollama(model=model_name_1)

# chain = create_sql_query_chain(llm, mtcars_db)
# response = chain.invoke({"question": "whats is count of cars"})
# # Whats the MPG Value for Merc 280
# logger.debug(f'-response-->> {response}')

"""

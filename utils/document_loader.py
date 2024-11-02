from langchain_community.document_loaders import (
                DirectoryLoader,
                PyPDFLoader,
                TextLoader,
            )
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

class ChromaCRUD:
    """
    """

    @classmethod
    def load_documents_into_database(self,
                                    model_name: str, 
                                    documents_path: str) -> Chroma:
                                
                                    """
                                    """
                                    logger.debug("-STARTED-load_documents_into_database ->>") 
                                    raw_documents = self.load_documents(documents_path)
                                    split_documents = TEXT_SPLITTER.split_documents(raw_documents)
                                    chroma_db_persist = Chroma.from_documents(split_documents, 
                                                                            OllamaEmbeddings(model=model_name),  
                                                                            persist_directory="./chroma_db", 
                                                                            collection_name='db_col_1')
                                    chroma_db_persist.persist()
                                    #docs = chroma_db_persist.similarity_search(query)
                                    logger.debug("-DONE-load_documents_into_database--TYPE-chroma_db_persist ->> %s" ,type(chroma_db_persist)) 

                                    ## below in memory ? 
                                    # db = Chroma.from_documents(
                                    #                 documents,
                                    #                 OllamaEmbeddings(model=model_name),
                                    #             )
                                    
                                    #import chromadb # TODO - https://python.langchain.com/docs/integrations/vectorstores/chroma/#initialization-from-client
                                    # persistent_client = chromadb.PersistentClient()
                                    # collection = persistent_client.get_or_create_collection("collection_name")
                                    # collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

                                    # vector_store_from_client = Chroma(
                                    #     client=persistent_client,
                                    #     collection_name="collection_name",
                                    #     embedding_function=embeddings,
                                    # )

                                    return chroma_db_persist

    @classmethod
    def get_docs_from_chromaDB(self):
        """
        """
        # db3 = Chroma(collection_name='db_col_1', persist_directory="./chroma_db", embedding_function)
        # docs = db3.similarity_search(query)
        # print(docs[0].page_content)

    @classmethod
    def load_documents(self,path: str) -> List[Document]:
        """
        Loads documents from the specified directory path.

        This function supports loading of PDF, Markdown, and HTML documents by utilizing
        different loaders for each file type. It checks if the provided path exists and
        raises a FileNotFoundError if it does not. It then iterates over the supported
        file types and uses the corresponding loader to load the documents into a list.

        Args:
            path (str): The path to the directory containing documents to load.

        Returns:
            List[Document]: A list of loaded documents.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")

        loaders = {
            ".pdf": DirectoryLoader(
                path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True,
            ),
            ".md": DirectoryLoader(
                path,
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True,
            ),
        }

        docs = []
        for file_type, loader in loaders.items():
            print(f"Loading {file_type} files")
            docs.extend(loader.load())
        return docs

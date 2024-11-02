import streamlit as st
import os
from langchain_community.llms import Ollama
from document_loader import ChromaCRUD 

from ollama_ingest_models import get_list_of_models
from ollama_llm import OLLAMA_LLM


from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

EMBEDDING_MODEL = "nomic-embed-text"
PATH = "pdf_dir"

logger.debug("--EMBEDDING_MODEL->> %s",type(EMBEDDING_MODEL))
st.title("ollama_langchain")
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()
    logger.debug("-init--st.session_state--->> %s",st.session_state)


selected_model_1 = st.sidebar.selectbox("Model to Download:", st.session_state["list_of_models"])

selected_model = st.sidebar.selectbox("Select a model:", st.session_state["list_of_models"])
if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = Ollama(model=selected_model)

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


if st.chat_input != None:
    logger.debug("-chat_input -NOT-NONE->>")

    if prompt := st.chat_input("Question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        #if Missed Creating Embeddings and DB Chroma 
        if "db" in st.session_state:
            logger.debug("-Got Session DB->> %s",type(st.session_state["db"]))
            st.info("INFO-Creating Embeddings and DB Chroma Done..")
        else:
            logger.debug("-Session DB-NONE-")
            st.warning("ERROR -- Missed Creating Embeddings and DB Chroma earlier.--This will take some time now.")
            st.session_state["db"] = ChromaCRUD().load_documents_into_database(
                                                        EMBEDDING_MODEL, 
                                                        folder_path
                                                    )
            st.info("[INFO] -You may now ask your Questions.")
            logger.debug("-Got Session DB--aa-->> %s",type(st.session_state["db"]))
            
            #TODO -- if DB had collection with same Name Dont Recreate COLL_NAME
            # collection = client.get_collection(name="test") 
            #with st.chat_message("assistant"):
            stream = OLLAMA_LLM.get_streaming_chain(st.session_state["llm"],
                                                st.session_state["db"],
                                                prompt,
                                                #st.session_state.messages,
                                                )
            logger.debug("-get_streaming_chain-STREAM->> %s",stream) ##<generator object RunnableSequence.stream at 0x7f078e30f100>
            response = st.write_stream(stream)
            logger.debug("-get_streaming_chain-response->> %s",type(response))
            st.session_state.messages.append({"role": "assistant", "content": response})

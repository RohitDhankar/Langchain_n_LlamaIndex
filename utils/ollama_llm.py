
## https://github.com/amscotti/local-LLM-with-RAG

from operator import itemgetter

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate

from utils.util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from .prompt_templates import (SYS_PROMPT_1,
                            REPHRASE_USER_QUERY,
                            LLM_RESPONSE_TEMPLATE)

SYS_PROMPT_TEMPLATE = PromptTemplate.from_template(SYS_PROMPT_1)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_USER_QUERY)
ANSWER_PROMPT = ChatPromptTemplate.from_template(LLM_RESPONSE_TEMPLATE)

logger.debug("----LOGGING--1--->> %s",type(SYS_PROMPT_TEMPLATE))

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
                        template="Source Document: {source}, Page {page}:\n{page_content}"
                    )


class OLLAMA_LLM:
    """
    Desc:
        - Ollama LLM Class

    """

    @classmethod
    def combine_documents(self,
                        docs, 
                        document_prompt=DEFAULT_DOCUMENT_PROMPT, 
                        document_separator="\n\n"):
        """
        Desc:
            - 
        """                   
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    @classmethod
    def get_streaming_chain(self,
                        llm, 
                        db,
                        question):
        """
        Desc:
            - get_streaming_chain
        """
        logger.debug("--get_streaming_chain--->>")
        logger.debug("---get_streaming_chain--Got Session DB->> %s",type(db)) 
        #<class 'langchain_community.vectorstores.chroma.Chroma'>
        conv_memory = ConversationBufferMemory(return_messages=True,
                                                output_key="answer",
                                                input_key="question")

        retriever = db.as_retriever(search_kwargs={"k": 10})

        # TODO -- Original Code ERRORS out 
        # loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(
        #                                         lambda x: "\n".join(
        #                                             [f"{item['role']}: {item['content']}" for item in x["memory"]]
        #                                         )
        #                                     ),
        #                                 )

        standalone_question = {
                            "standalone_question": {
                                "question": lambda x: x["question"],
                                "chat_history": lambda x: x["chat_history"],
                            }
                            | CONDENSE_QUESTION_PROMPT
                            | llm
                        }

        retrieved_documents = {
                                "docs": itemgetter("standalone_question") | retriever,
                                "question": lambda x: x["standalone_question"],
                            }

        final_inputs = {
                            "rag_docs_content": lambda x: self.combine_documents(x["docs"]),
                            "question": itemgetter("question"),
                        }

        answer = final_inputs | ANSWER_PROMPT | llm
        #final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        # final_chain = standalone_question | retrieved_documents | answer
        # obj_final_chain_stream = final_chain.stream({"question": question, "memory": conv_memory}) 
        #logger.debug("--RETURN-get_streaming_chain---obj_final_chain_stream->> %s",type(obj_final_chain_stream)) 
        logger.debug("--RETURN-get_streaming_chain---answer->> %s",type(answer)) 
        ##<class 'langchain_core.runnables.base.RunnableSequence'>
        return answer

    @classmethod
    def get_chat_chain(self,
                        llm, 
                        db):
        
        """
        Desc:
            - get_chat_chain
        """
        conv_memory = ConversationBufferMemory(
                return_messages=True, output_key="answer", input_key="question"
                    )

        retriever = db.as_retriever(search_kwargs={"k": 10})

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(conv_memory.load_memory_variables)
            | itemgetter("history"),
        )

        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
        }

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }

        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: self.combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }

        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs
            | ANSWER_PROMPT
            | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
            "docs": itemgetter("docs"),
        }

        final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        return final_chain

    @classmethod
    def chat(self,
            llm,
            db,
            question: str):
        """
        Desc:
            - 
        """

        final_chain = self.get_chat_chain(llm,db)
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"]})

        return chat

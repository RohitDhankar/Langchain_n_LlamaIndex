

SYS_PROMPT_1 = """

this is the SYS 

"""


REPHRASE_USER_QUERY = """Given the following conversation and a follow-up question, 
rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""


LLM_RESPONSE_TEMPLATE = """
### Instruction:
You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.
If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources

## Research:
{rag_docs_content}

## Question:
{question}
"""

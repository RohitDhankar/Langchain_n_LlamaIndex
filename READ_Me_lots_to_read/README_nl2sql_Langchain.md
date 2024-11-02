# Nl2Sql_Langchain

#

- whats the Tools within the - SQLDatabaseToolkit
- SQLDatabaseToolkit-tools is a LIST of Tools 
#

- QuerySQLDataBaseTool
- description="Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.", db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x7fcbb35bf860>), 

- InfoSQLDatabaseTool
- description='Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x7fcbb35bf860>), 

- ListSQLDatabaseTool(db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x7fcbb35bf860>), 

- QuerySQLCheckerTool
- description='Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!', db=<langchain_community.utilities.sql_database.SQLDatabase object at 0x7fcbb35bf860>, 

llm=Ollama(model='llama3.2'), 
llm_chain=LLMChain(verbose=False, prompt=PromptTemplate(input_variables=['dialect', 'query'], input_types={}, partial_variables={}, 
template='\n{query}\nDouble check the {dialect} query above for common mistakes, including:\n- Using NOT IN with NULL values\n- Using UNION when UNION ALL should have been used\n- Using BETWEEN for exclusive ranges\n- Data type mismatch in predicates\n- Properly quoting identifiers\n- Using the correct number of arguments for functions\n- Casting to the correct data type\n- Using the proper columns for joins\n\nIf there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.\n\nOutput the final SQL query only.\n\nSQL Query: '), 
llm=Ollama(model='llama3.2'), 
output_parser=StrOutputParser(), llm_kwargs={}))]



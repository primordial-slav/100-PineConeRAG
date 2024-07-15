from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
import pprint


class Prompt:
    def __init__(self,open_ai_model,pinecone_model,embedding_model,namespace):
        #self.pinecone_model = pinecone_model
        self.open_ai_model = open_ai_model
        self.embedding_model = embedding_model
        self.namespace = namespace
        self.messages = [
                        SystemMessage(content="You are my assistant and you are well versed in the Serbian language."),
                        #HumanMessage(content="Ћао, како си?"),
                        #AIMessage(content="Одлично сам, како ти могу помоћи данас?"),
                        ]
        
    def store_vectors(self,data,index_name,llm):
        docsearch = PineconeVectorStore.from_documents(
                                    documents=data,
                                    index_name=index_name,
                                    embedding=self.embedding_model, 
                                    namespace=self.namespace 
                                )
        self.qa = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=docsearch.as_retriever()
                            )

    #@staticmethod
    def augment_prompt(self,query: str):
        # get top 3 results from knowledge base
        results = self.qa.invoke(query)['result']
        # get the text from the results
        #source_knowledge = "\n".join([x.page_content for x in results])
        source_knowledge = results
        # feed into an augmented prompt
        augmented_prompt = f"""Koristeći sledeći kontekst odgovori na pitanje.

        Kontekst:
        {source_knowledge}

        Pitanje: {query}"""
        return augmented_prompt
    
    def get_response(self,query):
        # create a new user prompt
        prompt = HumanMessage(
            content=self.augment_prompt(query)
        )
        # add to messages
        self.messages.append(prompt)

        res = self.open_ai_model(self.messages)

        pprint.pprint(res.content)
        return res.content
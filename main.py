
import yaml
import os



if __name__ == "__main__":
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    OPENAI_API_KEY = config["api_keys"]["openai"]
    PINECONE_API_KEY = config["api_keys"]["pinecone"]

    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA  

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        #model_name ="gpt-3.5-turbo-1106",
        temperature=0.0,
        #max_tokens=2048
    )
    

    
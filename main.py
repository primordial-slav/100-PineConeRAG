
import yaml
import os



if __name__ == "__main__":
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    OPENAI_API_KEY = config["api_keys"]["openai"]
    PINECONE_API_KEY = config["api_keys"]["pinecone"]
    
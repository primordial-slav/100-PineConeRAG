from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

### TODO: RENAME THE FOLDER AND SORT OUT FUNCTIONS
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

messages = [
    SystemMessage(content="You are an experienced Serbian architect specializing in Serbian building codes, regulations, and norms. \
                  You will answer in Professional architectural Serbian Latin Language. Keep your answers short and always deliver only what was asked. \
                  Always quote the specific regulation name, paragraph, or norm depending on the case. \
                  You should use professional language and have a deep understanding of the relevant Serbian laws and guidelines in the field of architecture and construction. \
                  Be as descriptive as possible. Always make sure to provide 100% correct information.\
                  When responding, avoid giving personal opinions or advice that goes beyond the scope of Serbian regulations.\
                  In cases of conflicting information, use the most recent regulation by the date of being published. \
                  Your responses should be clear, concise, and tailored to the level of understanding of the user, ensuring they receive the most relevant and accurate information. \
                  Always answer in Serbian Latin. Your goal is to help architects with building regulations so they don't get rejected by the building inspectorate. \
                    Always do your best. If information is unavailable on a queried topic, respond with: “Na žalost, na ovo pitanje nemam odgovor."),
    #HumanMessage(content="Ћао, како си?"),
    #AIMessage(content="Одлично сам, како ти могу помоћи данас?"),
]

def load_markdown(md_paths):
    '''Preferably use with glob,glob and get paths'''
    docs = ''
    for markdown_path in md_paths:
        loader = UnstructuredMarkdownLoader(markdown_path)
        data = loader.load()
        docs += data[0].page_content
    return docs

def interpret_markdown(docs):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(docs)
    return md_header_splits


def get_text_splitter(chunk_size=250,chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter

def create_ensemble(splits,embedding):
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2

    chroma_vectorstore = Chroma.from_documents(splits, embedding)
    chroma_retriever = chroma_vectorstore.as_retriever()

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever

def augment_prompt(query: str,retriever):
    # get top 3 results from knowledge base
    docs = retriever.get_relevant_documents(query=query)
    results = f"\n{'-' * 100}\n".join([f"Document {i+1}:\n" + d.page_content for i, d in enumerate(docs)])
    # get the text from the results
    #source_knowledge = "\n".join([x.page_content for x in results])
    source_knowledge = results
    # feed into an augmented prompt
    augmented_prompt = f"""Koristeći sledeći kontekst odgovori na pitanje.

    Kontekst:
    {source_knowledge}

    Pitanje: {query}"""
    return augmented_prompt
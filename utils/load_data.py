from langchain_community.document_loaders import UnstructuredMarkdownLoader

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
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
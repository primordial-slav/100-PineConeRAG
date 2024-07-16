from langchain_community.document_loaders import UnstructuredMarkdownLoader

def load_markdown(md_paths):
    '''Preferably use with glob,glob and get paths'''
    docs = ''
    for markdown_path in md_paths:
        loader = UnstructuredMarkdownLoader(markdown_path)
        data = loader.load()
        docs += data[0].page_content
    return docs
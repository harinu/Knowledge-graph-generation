from langchain.text_splitter import RecursiveCharacterTextSplitter
from helper import Helper



class Document():
    def __init__(self, chunk_size=10000, chunk_overlap=300):
        hlp = Helper()
        self.log = hlp.get_logger()
        self.text_splitter = RecursiveCharacterTextSplitter(separators=[".\n"], 
                                                            chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    def chunk_documents(self, text):
        try: 
            chunked_document = self.text_splitter.create_documents([text])
            self.log.info("Successfully chunked the document!")
            return chunked_document
        except Exception as e:
            self.log.error("Error chunking the document!")
            return None
    

    def count_tokens(self, text):
        words = text.split()
        num_tokens = len(words)
        return num_tokens
                                                           

    
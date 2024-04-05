import os
from helper import Helper
from model import GeneratorModel
from docutils import Document


class GenerateData:
    def __init__(self):
        helper = Helper()
        helper.clear_log_file()
        self.log = helper.get_logger()

        self.model = GeneratorModel()
        self.doc = Document()

    
    def generate_predictions(self, text, whatfor = 'process'):
        if len(text) > 10000:
            chunks = self.doc.chunk_documents(text)
            processed_content = ''
            for chunk in chunks:
                self.log.info(f"Processing chunks")
                chunk_claude = self.model.model_prediction(chunk.page_content, whatfor)
                processed_content += chunk_claude+'\n'
        else:
            self.log.info(f"Processing text")
            processed_content = self.model.model_prediction(text, whatfor)
        
        return processed_content
    

    def get_txt_files(self, directory):
        return [f for f in os.listdir(directory) if f.endswith('.txt')]


    def read_data(self, file):
        with open(file, 'r') as f:
            data = f.read()
        return data
    

    def generate_training_data(self, directory):
        for file in self.get_txt_files(directory):
            data = self.read_data(f"{directory}/{file}")
            predictions = self.generate_predictions(data, whatfor = 'process')
            self.log.info(predictions)


if __name__ == '__main__':
    generate_data = GenerateData()
    generate_data.generate_training_data('data')
import os
import re
import pandas as pd
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
    

    def save_to_dataframe(self, data_list, existing_df=None):
        try:
            if existing_df is None:
                df = pd.DataFrame(columns=['id', 'input', 'output'])
            else:
                df = existing_df
            n = len(df)

            rows = []
            for i, item in enumerate(data_list):
                item_dict = {'id': n+i, 'input': item['input'], 'output': item['output']}
                rows.append(item_dict)
            
            new_rows = pd.DataFrame(rows)
            self.log.info(f"New Rows: {new_rows}")
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv('data.csv', index=False)
            self.log.info("Successfully saved to dataframe!")
            return True
        except Exception as e:
            self.log.error(f"Error saving to dataframe: {e}")
            return False
        
    
    
    def post_process_data(self, data):
        pattern = r"<sample>(.*?)</sample>"
        matches = re.findall(pattern, data, re.DOTALL)
        samples = []
    
        for match in matches:
            # match = match.strip().strip('\n')
            match = match.replace('\n', '')
            sample_dict = eval(match)
            samples.append(sample_dict)
        return samples 
        

    def generate_predictions(self, text, whatfor = 'process'):
        try:
            if len(text.split()) > 4000:
                chunks = self.doc.chunk_documents(text)
                processed_output = []
                for i, chunk in enumerate(chunks):
                    df = self.get_dataframe()
                    n = len(chunks)
                    self.log.info(f"Processing chunks {i+1}/{n}")
                    chunk_claude = self.model.model_prediction(chunk.page_content, whatfor)
                    self.log.info(f"Prediction: {chunk_claude}")
                    processed_output = self.post_process_data(chunk_claude)
                    self.log.info(f"Processed Output: {processed_output}")
                    self.save_to_dataframe(processed_output, df)
            else:
                df = self.get_dataframe()
                self.log.info(f"Processing text")
                claude = self.model.model_prediction(text, whatfor)
                processed_output = self.post_process_data(claude)
                self.save_to_dataframe(processed_output, df)

            return True
        
        except Exception as e:
            self.log.error(f"Error generating predictions: {e}")
            return False

    def get_txt_files(self, directory):
        return [f for f in os.listdir(directory) if f.endswith('.txt')]


    def read_data(self, file):
        with open(file, 'r') as f:
            data = f.read()
        return data
    
    def get_dataframe(self):
        return pd.read_csv('data.csv')

    def generate_training_data(self, directory):
        for i, file in enumerate(self.get_txt_files(directory)):
            if file not in ['10.txt', '9.txt']:
                continue
            print(file)
            data = self.read_data(f"{directory}/{file}")
            self.log.info(f"Processing File: {i+1}")
            was_success = self.generate_predictions(data, whatfor = 'process')
            if was_success:
                self.log.info(f"Successfully generated training data for file {i}!")
            else:
                self.log.error("Error generating training data!")

if __name__ == '__main__':
    generate_data = GenerateData()
    generate_data.generate_training_data('data')
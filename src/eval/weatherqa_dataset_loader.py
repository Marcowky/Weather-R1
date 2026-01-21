import os
import json
import copy
from torch.utils.data import Dataset

from src.utils.prompt import WEATHER_R1_PROMPT, DIRECT_PROMPT


class WeatherR1Dataset(Dataset):
    def __init__(self, weather_r1_path, image_rft_path="", split='test', prompt_type='original', exclude_category=None, language='cn'):
        self.weather_r1_path = weather_r1_path
        self.image_rft_path = image_rft_path
        self.language = language
        # Load data
        with open(self.weather_r1_path, 'r') as f:
            datas = json.load(f)
        if 'split' in datas[0]:
            self.data_list = [data for data in datas if data['split'] == split]
        else:
            self.data_list = datas
        # Exclude specific categories if provided
        if exclude_category:
            self.data_list = [data for data in self.data_list if data['category'] not in exclude_category]
        # Select prompt template
        if prompt_type == 'weather-r1':
            self.template = WEATHER_R1_PROMPT
        else: 
            self.template = DIRECT_PROMPT

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data_list[idx])
        data['prompt'] = self.build_prompt(self.data_list[idx])
        data['image_path'] = os.path.join(self.image_rft_path, data['image_path'])
        if 'answer' in data[self.language]:
            data['answer'] = data[self.language]['answer']
        return data
    
    def build_prompt(self, data):
        # Get question text
        question = data[self.language]['question']

        # Pull options if present
        if 'choices' in data[self.language]:
            choices = data[self.language]['choices']
            # Build the option prompt
            choices_list = []
            for key, item in choices.items():
                choices_list.append(f'{key}. {item}')
            choices_prompt = '\n'.join(choices_list)
            # Fill template with question and choices
            prompt = self.template.format(question=question, choices=choices_prompt)
        else:
            # If no options, treat it as free-form QA
            prompt = question

        return prompt
    
    def evaluate(self, predictions):
        correct = 0
        for idx, prediction in enumerate(predictions):
            if prediction == self.data_list[idx]['answer']:
                correct += 1
        return correct / len(predictions)

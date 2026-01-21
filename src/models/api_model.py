import json
import base64

from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class api_client:
    def __init__(self, api_name='qwen'):
        self.client = self.load_client(api_name)

    def load_client(self, api_name):
        with open("src/utils/api_config.json", "r") as f:
            data = json.load(f)
            config = data[api_name]
            api_key = config["api_key"]
            base_url = config["base_url"]

        return OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        
    def call_api_by_message(self, messages, model, temperature=1, return_json=False):
        if return_json:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        else:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
            )

        return chat_completion.choices[0].message.content


    def call_text_api(self, query, model="qwen-plus", temperature=1, return_json=False):
        messages = [{
            "role": "user",
            "content": query,
        }]
        return self.call_api_by_message(messages, model, temperature, return_json)


    def call_image_api(self, image_path, query, model="qwen-vl-max", temperature=1, return_json=False):
        base64_image = encode_image(image_path)
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}, 
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        return self.call_api_by_message(messages, model, temperature, return_json)


class APIModel:
    def __init__(self, api_name, model_name, **kwargs):
        self.api_client = api_client(api_name)
        self.model_name = model_name


    def generate(self, question, image_path, temperature=0, top_p=None, num_beams=1, max_new_tokens=1024):
        prompt = question

        output_text = self.api_client.call_image_api(image_path=image_path, 
                                                query=question, 
                                                model=self.model_name,
                                                temperature=temperature)

        return output_text, prompt
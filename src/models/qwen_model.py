import os
import re
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class QwenModel:
    def __init__(self, model_name, model_path, min_pixels=None, max_pixels=None):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model_path = os.path.expanduser(model_path)
        self.load_model()


    def load_model(self):
        
        if self.model_name.startswith("qwen2.5"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto"
            )
            # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #     self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
            # )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto"
            )
        
        processor_path = self.model_path + "-Instruct" if self.model_name.endswith("base") else self.model_path

        if self.min_pixels is not None and self.max_pixels is not None:
            self.processor = AutoProcessor.from_pretrained(processor_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        else:
            self.processor = AutoProcessor.from_pretrained(processor_path)


    @staticmethod
    def convert_messages(query, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        return messages


    @staticmethod
    def summarize_image_pad_token(prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        new_prompts = []
        
        # Collapse consecutive <|image_pad|> tokens into a counted shorthand
        for prompt in prompts:
            pattern = r'(<\|image_pad\|>)+'
            
            def replace_func(match):
                count = len(match.group(0)) // len('<|image_pad|>')
                return f'<|image_pad|*{count}>'
            
            new_prompt = re.sub(pattern, replace_func, prompt)
            new_prompts.append(new_prompt)
        
        return new_prompts


    def prepare_inputs(self, querys, image_paths):
        if isinstance(querys, str):
            querys = [querys]
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        messages = [
            QwenModel.convert_messages(query, image_path) for query, image_path in zip(querys, image_paths)
        ]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        return inputs, QwenModel.summarize_image_pad_token(texts)
    

    def generate(self, question, image_path, temperature=0, top_p=None, num_beams=1, max_new_tokens=1024):
        inputs, prompt = self.prepare_inputs(question, image_path)
        # Inference: Generation of the output
        if temperature > 0:
            generated_ids = self.model.generate(
                **inputs, 
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True)
        else:
            generated_ids = self.model.generate(
                **inputs, 
                do_sample=False,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True)            
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip(), prompt[0]


    def batch_generate(self, questions, image_paths, temperature=0, top_p=None, num_beams=1, max_new_tokens=512):
        inputs, prompts = self.prepare_inputs(questions, image_paths)
        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs, 
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts, prompts

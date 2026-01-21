# Required imports
import torch
import os

from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


class LLaVAModel:
    def __init__(self, model_name, model_path, **kwargs):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.context_len = None
        self.processor = None
        self.model_path = os.path.expanduser(model_path)
        self.load_model()


    def load_model(self):

        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        model_base = None

        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            self.model_path, model_base, model_name, attn_implementation="eager")


    def prepare_inputs(self, query, image, conv_mode='vicuna_v1'):
        qs = query
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Text prompt tokenization
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # Image preprocessing
        image_tensor = process_images([image], self.processor, self.model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()

        return prompt, input_ids, image_tensor


    def generate(self, question, image_path, temperature=0, top_p=None, num_beams=1, max_new_tokens=1024):
        image = Image.open(image_path).convert("RGB")
        prompt, input_ids, image_tensor = self.prepare_inputs(question, image)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True)

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return output_text, prompt

import os
import time
import json
import torch
# import google.generativeai as genai
# New API
from google import genai
from google.genai import types
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
from utils import setup_logger, convert_pdf_to_images, natural_sort_key, clean_json_string
import random

# Set up logging for gemini_agent
logger = setup_logger('llm_model', 'llm_model.log', level=logging.DEBUG)

class GeminiModel:
    def __init__(self,
                 prompt_template_dir="prompt_templates",
                 system_prompt_file="system.txt",
                 model_name="gemini-1.5-flash-002",
                 temperature=1.0,
                 top_p=0.95,
                 code_interpreter="inbuilt"):  # New argument added
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.code_interpreter = code_interpreter  # Store the interpreter choice
        self.client = self._initialize_model(prompt_template_dir, system_prompt_file)

    def _initialize_model(self, prompt_template_dir, system_prompt_file):
        # Set Gemini API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Get your API key from https://aistudio.google.com/app/apikey
        client = genai.Client(api_key=gemini_api_key)
        
        # Set generation configuration
        # generation_config = {
        #     "temperature": self.temperature,
        #     "top_p": self.top_p,
        #     # "top_k": 64,  # not avalable in gemini-1.5-pro
        #     "max_output_tokens": 8192,
        #     # "response_mime_type": "application/json",  # Some models do not support inbuilt code interpreter when response is in json mode.
        # }
        
        # Load system prompt
        if system_prompt_file is not None:
            with open(os.path.join(prompt_template_dir, system_prompt_file), 'r') as f:
                system_prompt = f.read()
        else:
            system_prompt = None

        # Determine the tools based on the code_interpreter argument
        if self.code_interpreter == "inbuilt":
            tools = 'code_execution'
        elif self.code_interpreter == "local":
            tools = []  # ['code_execution', execute_python_code] chat.send_message() does not support function calling yet. If use chat.send_message(), an "Invalid Argument" error will occur.
        elif self.code_interpreter == "cloud":
            tools = []  # Placeholder for future cloud support
        else:
            raise ValueError(f"Invalid code_interpreter value: {self.code_interpreter}")

        # Config
        generation_config = types.GenerateContentConfig(system_instruction=system_prompt, 
                                                        temperature=self.temperature,
                                                        top_p=self.top_p,
                                                        max_output_tokens=8192)
        self.gen_config = generation_config

        # Create the model
        response = client.models.generate_content(model=self.model_name, 
                                                  config=generation_config, 
                                                  contents=["Further instructions will be provided in the next message."])
        # model = genai.GenerativeModel(
        #     model_name=self.model_name,
        #     generation_config=generation_config,
        #     system_instruction=system_prompt,
        #     tools=tools,
        # )

        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"(temp, top_p) = ({self.temperature}, {self.top_p})")
        logger.info(f"tools: {tools}")
        logger.info(f"code_interpreter: {self.code_interpreter}")
        logger.debug(f"System prompt: {system_prompt}")

        return client

    def upload_to_gemini(self, path, mime_type=None):
        """Uploads the given file to Gemini.

        See https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        file = self.client.files.upload(file=path, config=dict(mime_type=mime_type))
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def wait_for_files_active(self, files):
        """Waits for the given files to be active.

        Some files uploaded to the Gemini API need to be processed before they can be
        used as prompt inputs. The status can be seen by querying the file's "state"
        field.

        This implementation uses a simple blocking polling loop. Production code
        should probably employ a more sophisticated approach.
        """
        logger.info("Waiting for file processing...")
        for name in (file.name for file in files):
            file = self.client.files.get(name=name)
            while file.state.name == "PROCESSING":
                logger.debug("File still processing...")
                time.sleep(10)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
        logger.info("All files ready")

    def query_pdf(self, pdf_path, query_prompt):

        # Upload the PDF file to Gemini
        files = [self.upload_to_gemini(pdf_path, mime_type="application/pdf")]

        # Some files have a processing delay. Wait for them to be ready.
        self.wait_for_files_active(files)

        logger.debug(f"Query prompt: {query_prompt}")

        # Generate content
        logger.info(f"Waiting for response from the model {self.model_name} ...")
        # chat = self.model.start_chat()
        response = self.client.models.generate_content(model=self.model_name, contents=[files[0], query_prompt], config=self.gen_config)
        # response = chat.send_message([files[0], query_prompt])
        logger.info(f"Response received from the model {self.model_name}.")

        try:
            cleaned_output_text = clean_json_string(response.text)
        except Exception as e:
            logger.error(f"Error cleaning JSON string: {e}")
            return {"error": "Invalid response", "raw_response": response}

        try:
            response_data = json.loads(cleaned_output_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Response text: {response.text}...")  # Log first 1000 characters
            return {"error": "Invalid JSON response", "raw_text": response.text}

        return response_data

    def generate_json_text(self, query_prompt):

        logger.debug(f"Generation prompt: {query_prompt}")

        # Generate content
        logger.info(f"Waiting for response from the model {self.model_name} ...")
        # chat = self.model.start_chat()
        response = self.client.models.generate_content(model=self.model_name, contents=[query_prompt], config=self.gen_config)
        # response = chat.send_message([files[0], query_prompt])
        logger.info(f"Response received from the model {self.model_name}.")

        try:
            cleaned_output_text = clean_json_string(response.text)
        except Exception as e:
            logger.error(f"Error cleaning JSON string: {e}")
            return {"error": "Invalid response", "raw_response": response}

        try:
            response_data = json.loads(cleaned_output_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Response text: {response.text}...")  # Log first 1000 characters
            return {"error": "Invalid JSON response", "raw_text": response.text}

        return response_data

class QwenVLModel:
    def __init__(self,
                 prompt_template_dir="prompt_templates",
                 system_prompt_file="system.txt",
                 model_name="Qwen/Qwen2-VL-7B-Instruct",
                 temperature=1.0,
                 top_p=0.95,
                 model_cache_dir=None,
                 debug_mode=False):
        self.prompt_template_dir = prompt_template_dir
        self.system_prompt_file = system_prompt_file
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.model_cache_dir = model_cache_dir
        self.debug_mode = debug_mode
        
        if not self.debug_mode:
            # Load the model
            self.model = self._load_model()

            # default processer
            self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
            # The default range for the number of visual tokens per image in the model is 4-16384.
            # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
            # min_pixels = 256*28*28
            # max_pixels = 1280*28*28
            # processor = AutoProcessor.from_pretrained(self.model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            print("Running in debug mode. Using dummy responses.")
        
        # Load system prompt
        if self.system_prompt_file is not None:
            with open(os.path.join(self.prompt_template_dir, self.system_prompt_file), 'r') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt=None

    def _load_model(self):
        # default: Load the model on the available device(s)
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        # )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=self.model_cache_dir
        )

        logger.info(f"Model loaded: {self.model_name}")

        return model

    def _generate_message_from_folder(self, folder_path, system_prompt=None, query_prompt=None):
        # Initialize the user content payload
        content = []

        # Get all image files from the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Sort the files using natural sorting
        image_files.sort(key=natural_sort_key)
        
        # Add images to content
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            content.append({
                "type": "image",
                "image": f"file://{os.path.abspath(image_path)}"
            })
        
        # Add text query to content
        content.append({
            "type": "text",
            "text": query_prompt
        })
        
        # Create the message structure
        if system_prompt is not None:
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt},
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        
        return messages

    def query_pdf(self, pdf_path, query_prompt):
        # Extract the pdf file name without the extension and use it to name the output directory
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        image_dir = os.path.join(os.path.dirname(pdf_path), pdf_name)

        # Convert the pdf to images
        num_pages = convert_pdf_to_images(pdf_path, image_dir)
        logger.info(f"Converted {num_pages} pages to images in {image_dir}")

        # Generate the prompt messages
        messages = self._generate_message_from_folder(image_dir, system_prompt=self.system_prompt, query_prompt=query_prompt)

        logger.debug(f"Query prompt: {json.dumps(messages, indent=2)}")
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        logger.info("Waiting for response from the model...")
        logger.debug(f"(temp, top_p) = ({self.temperature}, {self.top_p})")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=8192,  # Set the maximum number of new tokens to generate
            temperature=self.temperature,     # Set the temperature (0.0 to 1.0, lower is more deterministic)
            top_p=self.top_p,           # Set the top-p sampling parameter
            do_sample=True,      # Enable sampling (required for temperature and top_p to have an effect)
            # num_return_sequences=1,  # Number of alternative sequences to return
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info("Response received from the model.")

        cleaned_output_text = clean_json_string(output_text[0])

        try:
            return json.loads(cleaned_output_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Response text: {output_text[0][:1000]}...")  # Log first 1000 characters
            return {"error": "Invalid JSON response", "raw_text": output_text[0]}

    def query_images(self, image_dir, query_prompt):
        # Generate the prompt messages
        messages = self._generate_message_from_folder(image_dir, system_prompt=self.system_prompt, query_prompt=query_prompt)

        logger.debug(f"Query prompt: {json.dumps(messages, indent=2)}")
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        logger.info("Waiting for response from the model...")
        logger.debug(f"(temp, top_p) = ({self.temperature}, {self.top_p})")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=8192,  # Set the maximum number of new tokens to generate
            temperature=self.temperature,     # Set the temperature (0.0 to 1.0, lower is more deterministic)
            top_p=self.top_p,           # Set the top-p sampling parameter
            do_sample=True,      # Enable sampling (required for temperature and top_p to have an effect)
            # num_return_sequences=1,  # Number of alternative sequences to return
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info("Response received from the model.")

        cleaned_output_text = clean_json_string(output_text[0])

        try:
            return json.loads(cleaned_output_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            logger.error(f"Response text: {output_text[0][:1000]}...")  # Log first 1000 characters
            return {"error": "Invalid JSON response", "raw_text": output_text[0]}

    def _dummy_query_images(self, image_paths, query_prompt):
        # Choose a random dummy response file
        dummy_files = ['dummy_response_topic1.json', 'dummy_response_topic2.json']
        chosen_file = random.choice(dummy_files)
        
        # Load the dummy response
        with open(os.path.join('dummy_responses', chosen_file), 'r') as f:
            response = json.load(f)
        
        # Add some randomness to make it look more realistic
        # response["summary"] += f" The images provided cover pages {image_paths[0].split('_')[-1].split('.')[0]} to {image_paths[-1].split('_')[-1].split('.')[0]}."
        
        return json.dumps(response, indent=2)

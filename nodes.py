
import configparser
import io
import os
import numpy as np
import requests
import json
from PIL import Image
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_path = os.path.join(current_dir, "config.ini")
config = configparser.ConfigParser()
config.read(config_path)

class DeepInfraTextToImage:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,"default": "","tooltip": "The prompt"}),
                "model": (["black-forest-labs/FLUX-1.1-pro","black-forest-labs/FLUX-pro","black-forest-labs/FLUX-1-dev"],{"tooltip": "Model name to select"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,"tooltip": "The random seed used for creating the noise"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32,"tooltip": "Image width"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32,"tooltip": "Image height"}),
                "prompt_upsampling": ("BOOLEAN", {"default": False,"tooltip": "Whether to perform upsampling on the prompt"}),
                "safety_tolerance": ("INT", {"default": 6, "min": 0, "max": 6, "step": 1,"tooltip": "Tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "deepInfraTextToImage"
    CATEGORY = "YCYY/fluxapi"
    DESCRIPTION = "Generates an image using the DeepInfra API"
    def deepInfraTextToImage(self,prompt,model,seed,width,height,prompt_upsampling,safety_tolerance):
        try:
            api_key = config["API"]["DEEPINFRA_API_KEY"]
            if api_key == '':
                raise ValueError("api key is empty")
        except KeyError:
            raise ValueError("unable to find api key")
        if prompt == "":
            raise ValueError("Prompt is required")
        
        base_url = "https://api.deepinfra.com/v1/inference/"
        api_url = base_url + model
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "prompt_upsampling": prompt_upsampling,
            "safety_tolerance": safety_tolerance
        }
        timeout = 30 # timeout in seconds
        response = requests.post(api_url, headers=headers, data=json.dumps(payload),timeout=timeout)
        if response.status_code == 200:
            response_json  = response.json()
            print(response_json)
        else:
            raise ValueError("unable to generate image")
        image_url = response_json['image_url']
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0) 
            return (image,)
        except Exception as e:
            raise ValueError("unable to get image")

class SiliconFlowTextToImage:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,"default": "","tooltip": "The prompt"}),
                "model": (["black-forest-labs/FLUX.1-pro","black-forest-labs/FLUX.1-dev"],{"tooltip": "Model name to select"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 9999999999,"tooltip": "The random seed used for creating the noise"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32,"tooltip": "Image width"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32,"tooltip": "Image height"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 50,"step": 1,"tooltip": "Number of steps to generate the image"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.5, "max": 5,"step": 0.1,"tooltip": "Guidance scale.The higher this value is, the more the generated image tends to strictly match the content of the text prompt; the lower this value is, the more creative and diverse the generated image will be, and it may contain more unexpected elements."}),
                "prompt_upsampling": ("BOOLEAN", {"default": False,"tooltip": "Whether to perform upsampling on the prompt"}),
                "safety_tolerance": ("INT", {"default": 6, "min": 0, "max": 6, "step": 1,"tooltip": "Tolerance level for input and output moderation. Between 0 and 6, 0 being most strict, 6 being least strict"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "siliconFlowTextToImage"
    CATEGORY = "YCYY/fluxapi"
    DESCRIPTION = "Generates an image using the SiliconFlow API"
    def siliconFlowTextToImage(self,prompt,model,seed,width,height,steps,guidance,prompt_upsampling,safety_tolerance):
        try:
            api_key = config["API"]["SILICONFLOW_API_KEY"]
            if api_key == '':
                raise ValueError("api key is empty")
        except KeyError:
            raise ValueError("unable to find api key")
        if prompt == "":
            raise ValueError("Prompt is required")
        
        api_url = "https://api.siliconflow.cn/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "prompt_upsampling": prompt_upsampling,
            "safety_tolerance": safety_tolerance
        }
        if model == "black-forest-labs/FLUX.1-dev":
            payload = {
                "prompt": prompt,
                "model": model,
                "width": width,
                "height": height,
                "seed": seed,
                "num_inference_steps": steps,
                "prompt_upsampling": prompt_upsampling
            }
        timeout = 30 # timeout in seconds
        response = requests.post(api_url, headers=headers, data=json.dumps(payload),timeout=timeout)
        if response.status_code == 200:
            response_json  = response.json()
            print(response_json)
        else:
            raise ValueError("unable to generate image")
        image_url = response_json['images'][0]["url"]
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0) 
            return (image,)
        except Exception as e:
            raise ValueError("unable to get image")

NODE_CLASS_MAPPINGS = {
    "YCYY_DeepInfra_TextToImage": DeepInfraTextToImage,
    "YCYY_SiliconFlow_TextToImage": SiliconFlowTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCYY_DeepInfra_TextToImage": "YCYY DeepInfra TextToImage",
    "YCYY_SiliconFlow_TextToImage": "YCYY SiliconFlow TextToImage",
}   
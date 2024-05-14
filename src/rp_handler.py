import torch
import cv2
import math
import random
import numpy as np
import requests
import base64
import traceback

from PIL import Image, ImageOps

import diffusers

from pipeline_sdxl_photomaker import PhotoMakerStableDiffusionXLPipeline
from model_util import load_models_xl, get_torch_device, torch_gc

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger

from io import BytesIO
from huggingface_hub import hf_hub_download
from schemas.input import INPUT_SCHEMA
import os

# Global variables
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__('cuda') else torch.float32
DEFAULT_MODEL = 'wangqixun/YamerMIX_v8'


# Path to InstantID models
photomaker_path = f'./checkpoints/photomaker-v1.bin'

logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def load_image(image_file: str):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = load_image_from_base64(image_file)

    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    return image


def load_image_from_base64(base64_str: str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_bytes))
    return image


def determine_file_extension(image_data):
    image_extension = None

    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        else:
            # Default to png if we can't figure out the extension
            image_extension = '.png'
    except Exception as e:
        image_extension = '.png'

    return image_extension


def get_photomaker_instance(pretrained_model_name_or_path):
    if pretrained_model_name_or_path.endswith(
            '.ckpt'
    ) or pretrained_model_name_or_path.endswith('.safetensors'):
        scheduler_kwargs = hf_hub_download(
            repo_id='wangqixun/YamerMIX_v8',
            subfolder='scheduler',
            filename='scheduler_config.json',
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            scheduler_name=None,
            weight_dtype=dtype,
        )

        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
        pipe = PhotoMakerStableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            unet=unet,
            scheduler=scheduler,
        ).to(device)

    else:
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
        trigger_word="img"  # define the trigger word
    )     

    pipe.fuse_lora()

    return pipe


CURRENT_MODEL = DEFAULT_MODEL
PIPELINE = get_photomaker_instance(CURRENT_MODEL)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.Resampling.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image



def generate_image(
        job_id,
        model,
        face_image,
        prompt,
        negative_prompt,
        num_steps,
        guidance_scale,
        seed,
        width,
        height,
        start_merge_step
        ):

    global CURRENT_MODEL, PIPELINE

    if face_image is None:
        raise Exception(f'Cannot find any input face image! Please upload the face image')

    if prompt is None:
        prompt = 'a person'

    face_image_base64 = BytesIO(base64.b64decode(face_image))

    # Open the image using PIL
    image = Image.open(face_image_base64)

    input_id_images = []
    input_id_images.append(load_image(image))

    generator = torch.Generator(device=device).manual_seed(seed)

    logger.info('Start inference...', job_id)
    logger.info(f'Model: {model}', job_id)
    logger.info(f'Prompt: {prompt})', job_id)
    logger.info(f'Negative Prompt: {negative_prompt}', job_id)

    if model != CURRENT_MODEL:
        PIPELINE = get_photomaker_instance(model)
        CURRENT_MODEL = model
    
    images = PIPELINE(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        start_merge_step=start_merge_step,
        height=height,
        width=width,
        generator=generator
    ).images

    return images


def handler(job):
    try:
        validated_input = validate(job['input'], INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {
                'error': validated_input['errors']
            }

        payload = validated_input['validated_input']

        images = generate_image(
            job['id'],
            payload.get('model'),
            payload.get('face_image'),
            payload.get('prompt'),
            payload.get('negative_prompt'),
            payload.get('num_steps'),
            payload.get('guidance_scale'),
            payload.get('seed'),
            payload.get('width'),
            payload.get('height'),
            payload.get('start_merge_step'),
        )

        result_image = images[0]
        output_buffer = BytesIO()
        result_image.save(output_buffer, format='JPEG')
        image_data = output_buffer.getvalue()

        return {
            'image': base64.b64encode(image_data).decode('utf-8')
        }
    except Exception as e:
        logger.error(f'An exception was raised: {e}')

        return {
            'error': str(e),
            'output': traceback.format_exc(),
            'refresh_worker': True
        }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )


import torch

INPUT_SCHEMA = {
    'model': {
        'type': str,
        'required': False,
        'default': 'wangqixun/YamerMIX_v8'
    },
    'face_image': {
        'type': str,
        'required': True,
    },
    'prompt': {
        'type': str,
        'required': False,
        'default': 'a person'
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_steps': {
        'type': int,
        'required': False,
        'default': 30
    },
    'start_merge_step': {
        'type': int,
        'required': False,
        'default': 0.8
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 5
    },
    'seed': {
        'type': int,
        'required': False,
        'default': int(torch.seed()) % (2**32)
    },
    'width': {
        'type': int,
        'required': False,
        'default': 0
    },
    'height': {
        'type': int,
        'required': False,
        'default': 0
    }
}

import torch
from diffusers.models import ControlNetModel
from pipeline_sdxl_photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download


def fetch_photomaker_instances():
    """
    Fetches Photomaker checkpoints from the HuggingFace model hub.
    """
    hf_hub_download(
        repo_id="TencentARC/PhotoMaker", 
        local_dir='./checkpoints', 
        local_dir_use_symlinks=False, 
        filename="photomaker-v1.bin", 
        repo_type="model"
    )


def fetch_pretrained_model(model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return PhotoMakerStableDiffusionXLPipeline.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f'Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...')
            else:
                raise


def get_photomaker_pipeline():
    """
    Fetches the InstantID pipeline from the HuggingFace model hub.
    """
    torch_dtype = torch.float16

    args = {
        'torch_dtype': torch_dtype,
    }

    pipeline = fetch_pretrained_model('wangqixun/YamerMIX_v8', **args)

    return pipeline


if __name__ == '__main__':
    fetch_photomaker_instances()
    get_photomaker_pipeline()


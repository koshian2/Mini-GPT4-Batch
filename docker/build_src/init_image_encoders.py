from minigpt4.models.eva_vit import create_eva_vit_g
from minigpt4.common.dist_utils import download_cached_file

def init():
    create_eva_vit_g()
    download_cached_file("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")

if __name__ == "__main__":
    init()
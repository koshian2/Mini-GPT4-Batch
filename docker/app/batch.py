import sys
sys.path.append("/MiniGPT-4")

from minigpt4.conversation.conversation import Chat
from minigpt4.common.registry import registry
from minigpt4.common.config import Config
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2
import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.", default=os.environ["EVAL_CONFIG"])
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                'pretrain_llama2': CONV_VISION_LLama2}
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

    input_files = sorted(glob.glob("data/input/*"))
    prompt = "Discribe this image"
    for input_img in input_files:
        # upload image
        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = chat.upload_img(input_img, chat_state, img_list)

        # asking
        chat.ask(prompt, chat_state)

        # answer
        # num_beamsやtemperatureを変えたいときはこの引数をいじる
        llm_message = chat.answer(conv=chat_state,
                                img_list=img_list,
                                max_new_tokens=300,
                                max_length=2000)[0]
        
        # write outputs
        os.makedirs("data/output", exist_ok=True)
        input_basename = os.path.splitext(os.path.basename(input_img))[0]
        with open(f"data/output/{input_basename}.txt", "w", encoding="utf-8") as fp:
            fp.write(llm_message)
        
if __name__ == "__main__":
    main()
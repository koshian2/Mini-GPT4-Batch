import yaml
import copy
import os

def main():
    ## Convert environment.yml
    with open("environment.yml") as fp:
        data = yaml.safe_load(fp)
    for item in data["dependencies"]:
        if type(item) is dict:
            with open("requirements.txt", "w", encoding="utf-8") as fp:
                fp.write("\n".join(item["pip"]))

    ## Convert minigpt4/configs/models
    for config_path, model_name in zip(["minigpt4_llama2.yaml", "minigpt4_vicuna0.yaml"],
                                       ["llama2", "vicuna"]):
        path = f"minigpt4/configs/models/{config_path}"
        with open(path, "r") as fp:
            data = yaml.safe_load(fp)
            data["model"]["llama_model"] = f"/llm_weights/{model_name}"
        with open(path, "w") as fp:
            yaml.dump(data, fp)

    ## Convert eval_configs/minigpt4_eval.yaml
    # Vicuna 13B
    with open("eval_configs/minigpt4_eval.yaml", "r") as fp:
        orig = yaml.safe_load(fp)
    data = copy.deepcopy(orig)
    data["model"]["ckpt"] = "/minigpt4_ckpt/pretrained_minigpt4.pth"
    with open("eval_configs/minigpt4_vicuna_13b_eval.yaml", "w") as fp:
        yaml.dump(data, fp)
    # Vicuna 7B
    data = copy.deepcopy(orig)
    data["model"]["ckpt"] = "/minigpt4_ckpt/pretrained_minigpt4_7b.pth"
    with open("eval_configs/minigpt4_vicuna_7b_eval.yaml", "w") as fp:
        yaml.dump(data, fp)
    # LLaMA2 7B
    with open("eval_configs/minigpt4_llama2_eval.yaml", "r") as fp:
        data = yaml.safe_load(fp)
    data["model"]["ckpt"] = "pretrained_minigpt4_llama2_7b.pth"
    with open("eval_configs/minigpt4_llama2_eval.yaml", "w") as fp:
        yaml.dump(data, fp)
    os.remove("eval_configs/minigpt4_eval.yaml")
    
if __name__ == "__main__":
    main()

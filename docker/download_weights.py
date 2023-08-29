from huggingface_hub import snapshot_download
 
def main():
    local_root_dir = "G://LLMs/"

    snapshot_download(repo_id="Vision-CAIR/vicuna", local_dir=local_root_dir+"Vision-CAIR/vicuna", local_dir_use_symlinks=False)
    snapshot_download(repo_id="Vision-CAIR/vicuna-7b", local_dir=local_root_dir+"Vision-CAIR/vicuna-7b", local_dir_use_symlinks=False)
    snapshot_download(repo_id="meta-llama/Llama-2-7b-chat", local_dir=local_root_dir+"meta-llama/Llama-2-7b-chat", local_dir_use_symlinks=False)

if __name__ == "__main__":
    main()


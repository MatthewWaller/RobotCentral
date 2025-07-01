from huggingface_hub import snapshot_download

if __name__ == "__main__":
    local_dir = "./local_my_pi0"
    repo_id = "HelloCephalopod/my_pi0"
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded to: {local_dir}") 
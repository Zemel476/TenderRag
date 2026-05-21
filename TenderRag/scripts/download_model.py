# -*- coding: utf-8 -*-
"""
download_model.py


Author: shui-
Date: 2026/4/16 19:36
"""
import os
from huggingface_hub import snapshot_download

# 1. 强制使用国内镜像站，彻底绕开原来的网络报错
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 设置你要保存的本地绝对路径（你可以自己修改）
# 注意：Windows 路径里的反斜杠建议写两个 \\，或者在字符串前面加个 r
local_model_path = r"C:\cache\huggingface"
os.makedirs(local_model_path, exist_ok=True)


mode_dict = {
    # "BAAI/bge-base-zh-v1.5": "bge-base-zh-v1.5",
    # "infgrad/stella-base-zh-v3-1792d": "stella-base-zh",
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2",
    "moka-ai/m3e-base": "m3e-base",
}

for model_id, model_path in mode_dict.items():
    print(f"{model_id}: {model_path}")

    download_path = os.path.join(local_model_path, model_path)

    # 执行自动下载
    snapshot_download(
        repo_id=model_id, # 模型的名字
        local_dir=download_path,          # 下载到这个本地文件夹
        local_dir_use_symlinks=False,        # 不使用软链接，直接把实体文件下过来
        resume_download=True                 # 开启断点续传！万一断网了再跑一次就行
    )

    print(f"🎉 下载完成！模型已保存在: {download_path}")
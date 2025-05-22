# 离线部署流程
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-7B-Chat', revision='v1.1.4', local_dir='./qwen/Qwen-7B-Chat')  
print(model_dir)
model_dir = snapshot_download('Zhim puAI/chatglm3-3b', revision='v1.0.0', local_dir='./ZhipuAI/chatglm3-3b')

# 硬件加速配置
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True,
    use_flash_attn=True  # 开启FlashAttention加速
)
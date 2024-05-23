1. Install dependencies
```
python -m pip install -r requirements.txt
```
2. Download LLaVA models from Hugging Face
- [mmproj-mistral7b-f16-q6_k.gguf](https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/mmproj-mistral7b-f16-q6_k.gguf?download=true)
- [ggml-mistral-q_4_k.gguf](https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/ggml-mistral-q_4_k.gguf?download=true)

2. Connect UVC camera and run

```
python3 main.py
```

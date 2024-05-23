1. Install dependencies

```
python -m pip install -r requirements.txt

2. Deploy latest ion-kit binaries with LLM support

Build llama.cpp following instruction in `ion-kit/src/bb/llm/config.cmake`

Build ion-kit:
```
cmake -D CMAKE_BUILD_TYPE=Release -DLlama_DIR=<path-to-llama.cpp-install>/lib/cmake/Llama .. && cmake --build .
```

Replace binares:
```
cp ./install/lib/lib* <path-to-site-packages/ionpy/module/linux/
```


3. Download LLaVA models from Hugging Face

- [mmproj-mistral7b-f16-q6_k.gguf](https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/mmproj-mistral7b-f16-q6_k.gguf?download=true)
- [ggml-mistral-q_4_k.gguf](https://huggingface.co/cmp-nct/llava-1.6-gguf/resolve/main/ggml-mistral-q_4_k.gguf?download=true)

4. Connect UVC camera and run

```
python3 main.py
```

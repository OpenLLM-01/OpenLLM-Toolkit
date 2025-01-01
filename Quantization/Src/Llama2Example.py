# Variables
MODEL_ID = "mlabonne/EvolCodeLlama-7b"
QUANTIZATION_METHODS = ["q4_k_m", "q5_k_m"]

# Constants
MODEL_NAME = MODEL_ID.split('/')[-1]

# Install llama.cpp
!git clone https://github.com/ggerganov/llama.cpp
!cd llama.cpp && git pull && make clean && LLAMA_CUBLAS=1 make
!pip install -r llama.cpp/requirements.txt

# Download model
!git lfs install
!git clone https://huggingface.co/{MODEL_ID}

# Convert to fp16
fp16 = f"{MODEL_NAME}/{MODEL_NAME.lower()}.fp16.bin"
!python llama.cpp/convert.py {MODEL_NAME} --outtype f16 --outfile {fp16}

# Quantize the model for each method in the QUANTIZATION_METHODS list
for method in QUANTIZATION_METHODS:
    qtype = f"{MODEL_NAME}/{MODEL_NAME.lower()}.{method.upper()}.gguf"
    !./llama.cpp/quantize {fp16} {qtype} {method}


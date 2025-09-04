import torch
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Quantized model path
model_path = "C:/Users/priya/My_files/My_Projects/gpt_oss_20b_8gb/gpt-oss-20b-6gb-quantized-unsloth"

# 4-bit quantization config with offloading support
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True  # Explicitly allow 32-bit offload
)

# Custom device_map for GPT-NeoX-20B (minimal GPU usage)
custom_device_map = {}
for i in range(44):
    if i < 2 or i > 41:  # Only first 2 and last 2 layers on GPU (~4 layers, ~3-4 GB)
        custom_device_map[f"gpt_neox.layers.{i}"] = "cuda:0"
    else:
        custom_device_map[f"gpt_neox.layers.{i}"] = "cpu"
custom_device_map["gpt_neox.embed_in"] = "cpu"
custom_device_map["gpt_neox.embed_out"] = "cpu"
custom_device_map["gpt_neox.final_layer_norm"] = "cuda:0"  # Critical on GPU

try:
    # Load model with Transformers, then optimize with Unsloth
    logger.info("Loading quantized model with Transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map=custom_device_map,
        max_memory={0: "6000MiB", "cpu": "10000MiB"},
        torch_dtype=torch.float16
    )

    # Apply Unsloth optimizations
    logger.info("Applying Unsloth optimizations...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        model=model,  # Pass pre-loaded model
        tokenizer_name=model_path,
        device_map=custom_device_map,
        max_memory={0: "6000MiB", "cpu": "10000MiB"}
    )

    # Fallback to assign devices to unallocated parameters
    logger.info("Assigning devices to unallocated parameters...")
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            param.data = param.data.to("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        logger.info("Setting padding token to <|endoftext|>...")
        tokenizer.pad_token = "<|endoftext|>"

    # Generate text
    logger.info("Generating text...")
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    logger.error(f"Error during inference: {str(e)}")
    raise
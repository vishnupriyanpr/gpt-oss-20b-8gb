import torch
from unsloth import FastLanguageModel
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Model and output paths
model_name = "C:/Users/priya/My_files/My_Projects/gpt_oss_20b_8gb/gpt-oss-20b"
output_dir = "C:/Users/priya/My_files/My_Projects/gpt_oss_20b_8gb/gpt-oss-20b-6gb-quantized-unsloth"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

try:
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
        max_memory={0: "6500MiB", "cpu": "8000MiB"}
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        logger.info("Setting padding token to <|endoftext|>...")
        tokenizer.pad_token = "<|endoftext|>"

    # Save quantized model
    logger.info("Saving quantized model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Quantized model saved to {output_dir}")

except Exception as e:
    logger.error(f"Error during quantization: {str(e)}")
    raise
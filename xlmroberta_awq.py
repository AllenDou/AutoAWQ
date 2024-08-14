from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'BAAI/bge-reranker-base'
quant_path = '/root/bge-reranker-base-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
#quant_config = { "zero_point": False, "q_group_size": 32, "w_bit": 4, "version": "marlin" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

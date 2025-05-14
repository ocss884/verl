# test_sglang_server_spmd.py

import os
import torch
from sglang.srt.entrypoints.verl_engine import VerlEngine
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from verl.utils.torch_functional import pad_sequence_to_length

from utils_sglang import (
    levenshtein,
    are_lists_similar,
    initialize_global_process_group,
)

def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor):
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()

def test_sglang_server_spmd():
    # 1. Ensure at least two GPUs
    assert torch.cuda.device_count() >= 2

    # 2. Initialize distributed group (TP + DP)
    initialize_global_process_group()

    # 3. HF original model
    local_model_path = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare prompts
    prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name?",
    ]
    toks = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = pad_sequence_to_length(toks["input_ids"], 16, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(toks["attention_mask"], 16, 0, left_pad=True)

    # HF sync generation
    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path).bfloat16().cuda()
    gen_cfg = GenerationConfig(do_sample=False)
    hf_out = actor_model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        max_new_tokens=16,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=gen_cfg,
        use_cache=False,
        return_dict_in_generate=True,
    )
    hf_tokens = tokenizer.batch_decode(hf_out.sequences[:, 16:])

    # 4. Build DeviceMesh (CPU only manages parallelism, no inference)
    mesh = init_device_mesh(
        "cpu",
        mesh_shape=(1, 4, 1),
        mesh_dim_names=["dp", "tp", "pp"],
    )["tp"]
    tp_size = mesh.size()
    tp_rank = mesh.get_local_rank()

    # 5. Clean up TORCHELASTIC environment variables (avoid HTTPServerAdapter conflicts)
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        os.environ.pop(k, None)

    # Assign a unique port for each potential server instance
    # Starting from 30001 to avoid the default 30000 if it's stuck
    base_port = 30001
    server_port = base_port + tp_rank
    print(f"[Rank {tp_rank}] Attempting to use port: {server_port}")

    # 6. Instantiate VerlEngine and specify server backend
    llm = VerlEngine(
        model_path=local_model_path,
        dtype="bfloat16",
        device_mesh_cpu=mesh,
        base_gpu_id=0,
        gpu_id_step=1,
        backend="server",
        port=server_port,  # Pass the unique port
        nnodes=1,
    )

    # 7. Construct pure token list
    batch = input_ids.cuda()
    pad_id = tokenizer.pad_token_id
    idx_list = [ _pre_process_inputs(pad_id, batch[i]) for i in range(batch.size(0)) ]

    # 8. SGLang Server generation
    sampling_params = dict(
        n=1, temperature=0, top_p=1, top_k=-1,
        max_new_tokens=16,
        presence_penalty=0, frequency_penalty=0,
        repetition_penalty=1,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        ignore_eos=False,
    )
    outputs = llm.generate(input_ids=idx_list, sampling_params=sampling_params)

    # 9. Extract text and compare
    sglang_tokens = [ out["text"] for out in outputs ]
    print("HF outputs:   ", hf_tokens)
    print("SGLang server:", sglang_tokens)
    assert are_lists_similar(hf_tokens, sglang_tokens), "Difference >10%"

    # 10. Clean up
    llm.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    test_sglang_server_spmd()
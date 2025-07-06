import os
import time
import torch

from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from sglang.srt.entrypoints.engine import Engine

from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.utils import broadcast_pyobj, get_ip, MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.server_args import PortArgs, ServerArgs
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils.fsdp_utils import init_fn, get_fsdp_wrap_policy
from time import sleep


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor

def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    # NOTE MODIFIED should provide backend=None to have nccl+gloo
    # torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend=f"cpu:gloo,cuda:nccl", timeout=timedelta(seconds=timeout_second))


    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def main():
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    # torch.cuda.memory._record_memory_history()
    assert torch.cuda.is_available(), "CUDA not available"
    print("Start init")
    local_rank, rank, world_size = initialize_global_process_group()
    torch.cuda.set_device(local_rank)
    print(f"RANK{rank}", local_rank, rank, world_size)
    
    "============================================= Parallel ============================================="
    # 1
    # dp, tp, pp = 1, 16, 1
    # device_count = 8
    # model_name = "deepseek-ai/DeepSeek-V3"
    # 2
    dp, tp, pp = 1, 8, 1
    device_count = 8
    model_name = "Qwen/Qwen3-30B-A3B"
    # 3
    # dp, tp, pp = 2, 4, 1
    # device_count = 4
    # model_name = "Qwen/Qwen2-7B-Instruct"
    # 4
    # dp, tp, pp = 2, 8, 1
    # device_count = 8
    # model_name = "Qwen/Qwen2.5-32B-Instruct"
    # 5
    # dp, tp, pp = 1, 8, 1
    # device_count = 8
    # model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    "============================================= Parallel ============================================="
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    "============================================= FSDP ============================================="
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(dp*tp,), mesh_dim_names=["fsdp"]
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    print(f"RANK{rank}: loaded model")

    auto_wrap = get_fsdp_wrap_policy(actor_model, None, False)
    fsdp_model = FSDP(
        actor_model,
        cpu_offload=CPUOffload(offload_params=True),
        param_init_fn=init_fn,
        use_orig_params=True,
        auto_wrap_policy=auto_wrap,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        sync_module_states=False,
        device_mesh=device_mesh,
    )
    "============================================= FSDP ============================================="
    
    "============================================= INIT PARALLEL ============================================="
    kwargs = dict(
        mesh_shape=(dp, tp, pp), 
        mesh_dim_names=["dp", "tp", "pp"]
    )
    inference_device_mesh_cpu = init_device_mesh("cpu", **kwargs)

    "============================================= REMOVE FOR TORCHRUN ============================================="
    #NOTE: otherwise would fail
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    "============================================= REMOVE FOR TORCHRUN ============================================="
    
    "============================================= INIT PARALLEL ============================================="

    print("Building SGLang...")
    sleep(10)
    if rank == 0:
        llm = Engine(
            model_path=model_name,
            trust_remote_code=True,
            enable_deepep_moe=True,
            enable_dp_attention=True,
            deepep_mode="low_latency",
            enable_eplb=True,
            eplb_rebalance_num_iterations=1000,
            ep_dispatch_algorithm="static",
            expert_distribution_recorder_buffer_size=50,
            expert_distribution_recorder_mode="stat",
            tp_size=8,
            dp_size=8,
            nnodes=1,
            mem_fraction_static=0.5,
            # ep_num_redundant_experts=16,
            enable_dp_lm_head=True,
            dtype="bfloat16",
            base_gpu_id=0,
            gpu_id_step=1,
            enable_memory_saver=True,
            attention_backend="flashinfer",
            moe_dense_tp_size=1,
            
            cuda_graph_max_bs=128
        )
    torch.distributed.barrier(inference_device_mesh_cpu["tp"].get_group())

    if rank == 0:
        print('call release_memory_occupation', flush=True)
        llm.release_memory_occupation()
        print('sleep...', flush=True)
        time.sleep(3)
        print('call resume_memory_occupation', flush=True)
        llm.resume_memory_occupation()

    for tensor_index, (name, tensor) in enumerate(fsdp_model.state_dict().items()):
        serialized_tensor = MultiprocessingSerializer.serialize(
            _preprocess_tensor_for_update_weights(tensor)
        )

        if rank == 0:
            gathered_serialized_tensors = [None for _ in range(tp)]
        else:
            gathered_serialized_tensors = None
        torch.distributed.gather_object(
            obj=serialized_tensor,
            object_gather_list=gathered_serialized_tensors,
            dst=inference_device_mesh_cpu["tp"].mesh.tolist()[0],
            group=inference_device_mesh_cpu["tp"].get_group(),
        )

        if rank == 0:
            llm.update_weights_from_tensor(
                named_tensors=[
                    (
                        name,
                        LocalSerializedTensor(values=gathered_serialized_tensors),
                    )
                ],
                load_format=None,
                flush_cache=False,
            )
        torch.distributed.barrier(inference_device_mesh_cpu["tp"].get_group())
    if rank == 0:
        print("Weights updated successfully", flush=True)
    torch.distributed.barrier(inference_device_mesh_cpu["tp"].get_group())
    # print("updating rollout weights")
    # "============================================= GEN ============================================="
    sampling_params = dict(
        temperature=0, top_p=1, n=1, max_new_tokens=16, ignore_eos=True
    )

    if inference_device_mesh_cpu["tp"].get_local_rank() == 0:
        outputs = llm.generate("who are you", sampling_params=sampling_params)
        print("="*64)
        print(f'SGlang response: {outputs["text"]}')
        print("="*64)
    "============================================= GEN ============================================="
    

# Due to resouce constraints, I mimic the essential procedure RL needs between actor and rollout on a single H20 node.
# launch actor -> launch rollout -> release rollout memory -> (skip training) -> resume rollout memory -> generate to check if works

# torchrun --nnodes=1 --nproc_per_node=8 --master_addr=<YOUR IP> --master_port=34567 --node_rank 0 test_30b.py

if __name__ == "__main__":
    main()
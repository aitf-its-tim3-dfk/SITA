import argparse
import json
import os
import torch
import math
from safetensors.torch import load_file, save_file

def slerp(t, v0, v1, dot_threshold=0.9995):
    """Spherical linear interpolation between two tensors."""
    v0_norm = v0 / v0.norm()
    v1_norm = v1 / v1.norm()
    
    dot = (v0_norm * v1_norm).sum()
    
    if dot.isnan() or dot > dot_threshold:
        return torch.lerp(v0, v1, t)
    
    if dot < 0.0:
        v1_norm = -v1_norm
        dot = -dot
        v1 = -v1
        
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    return s0 * v0 + s1 * v1


def multislerp_tensor(tensors: list[torch.Tensor], weights: list[float], eps: float = 1e-8):
    """
    Implements barycentric interpolation on a hypersphere.
    Adapted from mergekit's multislerp.
    """
    if len(tensors) == 1:
        return tensors[0]

    tensors_stack = torch.stack(tensors, dim=0)
    tensors_flat = tensors_stack.view(tensors_stack.shape[0], -1)

    weights_tensor = torch.tensor(weights, dtype=tensors_stack.dtype, device=tensors_stack.device)
    weights_tensor = weights_tensor / weights_tensor.sum()

    # Project to unit hypersphere
    norms = torch.norm(tensors_flat, dim=-1, keepdim=True)
    unit_tensors = tensors_flat / (norms + eps)

    mean = (unit_tensors * weights_tensor.view(-1, 1)).sum(0)
    mean_norm = torch.norm(mean)
    
    if mean_norm < eps:
        # Fallback to linear interpolation (e.g. if interpolating towards a 0-tensor)
        res = torch.zeros_like(tensors_flat[0])
        for i in range(tensors_stack.shape[0]):
            res += tensors_flat[i] * weights_tensor[i]
        return res.view(tensors_stack.shape[1:])
        
    mean = mean / mean_norm

    # Project to tangent space
    dots = (unit_tensors * mean).sum(-1, keepdim=True)
    tangent_vectors = unit_tensors - dots * mean

    # Interpolate
    tangent_result = (tangent_vectors * weights_tensor.view(-1, 1)).sum(0)

    # Project back to sphere using exponential map
    tangent_norm = torch.norm(tangent_result) + eps
    result = mean * torch.cos(tangent_norm) + tangent_result * (torch.sin(tangent_norm) / tangent_norm)

    avg_norm = (norms.squeeze(-1) * weights_tensor).sum()
    result = result * avg_norm
    return result.view(tensors_stack.shape[1:])


def normalize_key(k):
    k_stripped = k
    while k_stripped.startswith("base_model.") or k_stripped.startswith("model."):
        if k_stripped.startswith("base_model."):
            k_stripped = k_stripped[len("base_model."):]
        elif k_stripped.startswith("model."):
            k_stripped = k_stripped[len("model."):]
    return k_stripped

def denormalize_key(k_stripped):
    return "base_model.model.model." + k_stripped

def merge_adapters(paths, out_path, weights):
    cfgs = []
    scales = []
    models = []
    
    for path in paths:
        with open(os.path.join(path, "adapter_config.json")) as f:
            cfg = json.load(f)
            cfgs.append(cfg)
        scale = cfg.get("lora_alpha", 16) / cfg.get("r", 16)
        scales.append(scale)
        
        t_raw = load_file(os.path.join(path, "adapter_model.safetensors"))
        t_norm = {normalize_key(k): v for k, v in t_raw.items()}
        models.append(t_norm)
        
    for i, (path, w, scale) in enumerate(zip(paths, weights, scales)):
        print(f"Adapter {i+1} (weight {w:.2f}) scale: {scale}")
        
    keys = set()
    for t in models:
        keys |= set(t.keys())
        
    print(f"Found {len(keys)} union tensors.")
    
    out_tensors = {}
    for k in keys:
        # Find shape and dtype from the first model that has this key
        ref_tensor = next(m[k] for m in models if k in m)
        
        tensors_for_k = []
        for v, scale in zip(models, scales):
            if k in v:
                tensors_for_k.append(v[k].float() * math.sqrt(scale))
            else:
                tensors_for_k.append(torch.zeros_like(ref_tensor, dtype=torch.float32))
            
        if len(paths) == 2:
            out_tensors[denormalize_key(k)] = slerp(weights[1], tensors_for_k[0], tensors_for_k[1]).to(ref_tensor.dtype)
        else:
            out_tensors[denormalize_key(k)] = multislerp_tensor(tensors_for_k, weights).to(ref_tensor.dtype)
            
    os.makedirs(out_path, exist_ok=True)
    save_file(out_tensors, os.path.join(out_path, "adapter_model.safetensors"))
    
    # Save config
    out_cfg = cfgs[0].copy()
    out_cfg["lora_alpha"] = out_cfg["r"]  # Set scale to 1.0
    
    # Union target_modules
    out_target_modules = set()
    for cfg in cfgs:
        tm = cfg.get("target_modules")
        if isinstance(tm, list):
            out_target_modules.update(tm)
            
    if out_target_modules:
        out_cfg["target_modules"] = list(out_target_modules)
        
    with open(os.path.join(out_path, "adapter_config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
        
    print(f"Saved merged adapter to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", nargs="+", required=True, help="List of adapter paths")
    parser.add_argument("--weights", nargs="+", type=float, required=True, help="List of weights")
    parser.add_argument("--out", required=True, help="Path to output adapter")
    args = parser.parse_args()
    
    assert len(args.adapters) == len(args.weights), "Must have equal number of adapters and weights"
    merge_adapters(args.adapters, args.out, args.weights)

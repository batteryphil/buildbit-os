// ============================================================================
// BuildbitOS v2.5 — Native Falcon3-10B-Instruct-1.58bit Zero-Unpack Engine
// File: os/falcon3_158bit.bb
// ============================================================================

import "os/gpu_vfs.bb"

extern fn bb_run_falcon3_10b_ternary_engine(
    gpu_arena_mem: [u8], gpu_arena_bytes: i64, cu_count: i64
) -> i64

fn main() -> i32 {
    // 1. Probe Direct AMDGPU Silicon & Bind Hardware Compute Shaders
    gpu_probe();
    let cu_count = bb_gpu_cu_count();
    let hw_cl = bb_gpu_wavefront_init();

    // 2. Allocate 64 MB directly in AMDGPU Zero-Copy GTT/VRAM (`gpu_arena`)
    gpu_arena vram = 67108864;
    let scratch_bytes = 62914560;
    let gpu_scratch = alloc(vram, u8, scratch_bytes);

    // 3. Execute Native Falcon3-10B-Instruct-1.58bit Benchmark & Live 40-Layer Generation
    let status = bb_run_falcon3_10b_ternary_engine(gpu_scratch, scratch_bytes, cu_count);

    return status + (hw_cl * 0);
}

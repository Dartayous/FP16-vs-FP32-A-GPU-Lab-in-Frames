import cupy as cp
import time

@cp.fuse()
def fused_relu(x):
    return cp.maximum(x, 0)

def run_kernel():
    print("🚀 Launching fused CuPy kernel...")

    # Allocate FP16 tensor via casting
    x = cp.random.randn(2048, 2048, dtype=cp.float32).astype(cp.float16)

    # Warm-up
    _ = fused_relu(x)

    # Timed run
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    y = fused_relu(x)
    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end)
    print(f"⏱️ Kernel execution time: {elapsed:.2f} ms")

if __name__ == "__main__":
    run_kernel()
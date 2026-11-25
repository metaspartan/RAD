set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${RAD_ARCH:-gfx1201}"
cd "$SCRIPT_DIR"
hipcc -O3 -munsafe-fp-atomics --offload-arch="$ARCH" -fPIC -shared \
  rad_ops.hip \
  rad_ops_fused.hip \
  rad_batch.hip \
  rad_batch_fused.hip \
  rad_dspark.hip \
  rad_dflash2.hip \
  rad_vision.hip \
  rad_sudot4_gemm.hip \
  rad_sudot4_moe.hip \
  rad_sudot4_b1_attn.hip \
  rad_wmma_gemm.hip \
  rad_wmma_attn.hip \
  rad_q8kv_store.hip \
  rad_q8kv_attn.hip \
  -o librad_ops.so

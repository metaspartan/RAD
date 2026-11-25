#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
ARCH="${RAD_CUDA_ARCH:-sm_121}"
NVCC="${CUDA_PATH}/bin/nvcc"
OUT="${SCRIPT_DIR}/librad_cuda_ops.so"

gcc -O3 -fPIC -march=armv8.2-a+fp16 -c "${SCRIPT_DIR}/rad_rdma.c" -o "${SCRIPT_DIR}/rad_rdma.o"

"${NVCC}" -O3 -arch="${ARCH}" --use_fast_math --shared -Xcompiler -fPIC \
    -I"${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/rad_exl3.cu" \
    "${SCRIPT_DIR}/rad_exl3_mma.cu" \
    "${SCRIPT_DIR}/rad_dsv41_ops.cu" \
    "${SCRIPT_DIR}/rad_dsv41_router.cu" \
    "${SCRIPT_DIR}/rad_rdma.o" \
    -libverbs \
    -o "${OUT}"
rm -f "${SCRIPT_DIR}/rad_rdma.o"
echo "Built ${OUT} for ${ARCH}"

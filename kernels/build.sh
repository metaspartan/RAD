set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${RAD_ARCH:-gfx1201}"
cd "$SCRIPT_DIR"
hipcc -O3 --offload-arch="$ARCH" -fPIC -shared \
  rad_ops.hip \
  rad_batch.hip \
  rad_dspark.hip \
  rad_sudot4.hip \
  rad_wmma.hip \
  -o librad_ops.so

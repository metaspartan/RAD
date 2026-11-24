set -euo pipefail
VER="${1:-0.1.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="$ROOT/dist/deb_stage"
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/usr/bin" "$PKG/usr/lib/rad" "$PKG/usr/share/doc/rad"
cp "$ROOT/target/release/rad" "$PKG/usr/bin/rad"
cp "$ROOT/kernels/librad_ops.so" "$PKG/usr/lib/rad/librad_ops.so"
cp "$ROOT/LICENSE" "$PKG/usr/share/doc/rad/copyright"
chmod 755 "$PKG/usr/bin/rad"
cat << EOF > "$PKG/DEBIAN/control"
Package: rad
Version: ${VER}
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Carsen Klock <metaspartan@users.noreply.github.com>
Description: RAD - High-performance Rust + ROCm/HIP multimodal LLM inference engine
 High-throughput LLM inference engine with native DFlash 2 speculative decoding,
 RDNA4 WMMA matrix core kernels, continuous batching, and GPU vision support.
EOF
mkdir -p "$ROOT/dist"
dpkg-deb --build "$PKG" "$ROOT/dist/rad_${VER}_amd64.deb"
rm -rf "$PKG"

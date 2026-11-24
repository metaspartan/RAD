class Rad < Formula
  desc "High-performance Rust + ROCm/HIP multimodal LLM inference engine"
  homepage "https://github.com/metaspartan/RAD"
  url "https://github.com/metaspartan/RAD/archive/refs/tags/v0.1.0.tar.gz"
  license "MIT"
  head "https://github.com/metaspartan/RAD.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "bash", "kernels/build.sh"
    system "cargo", "build", "--release", "--bin", "rad"
    lib.install "kernels/librad_ops.so"
    bin.install "target/release/rad"
  end

  test do
    assert_match "RAD - High-performance", shell_output("#{bin}/rad --help")
  end
end

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.parent().unwrap().parent().unwrap().join("kernels");
    println!("cargo:rustc-link-search=native={}", kernels_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", kernels_dir.display());
    println!("cargo:rerun-if-changed={}", kernels_dir.join("librad_ops.so").display());
}

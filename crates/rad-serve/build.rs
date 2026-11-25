use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.parent().unwrap().parent().unwrap().join("kernels").join("rocm");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", kernels_dir.display());
}

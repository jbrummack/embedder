#[cfg(feature = "burn")]
use burn_import::onnx::ModelGen;

fn main() {
    #[cfg(feature = "burn")]
    ModelGen::new()
        .input("src/model/efficientnet.onnx")
        .out_dir("model/")
        .run_from_script();
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos"))]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    uniffi::generate_scaffolding("src/embedder.udl").unwrap();
}

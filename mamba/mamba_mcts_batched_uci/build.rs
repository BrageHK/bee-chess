use std::path::PathBuf;
use std::process::Command;

fn libtorch_lib_dir() -> Option<PathBuf> {
    if std::env::var("LIBTORCH_USE_PYTORCH").is_ok() {
        let python = std::env::var("MAMBA_VENV_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let output = Command::new(&python)
            .args(["-c", "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        return Some(PathBuf::from(String::from_utf8(output.stdout).ok()?.trim()));
    }
    let libtorch = std::env::var("LIBTORCH").ok()?;
    Some(PathBuf::from(libtorch).join("lib"))
}

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=MAMBA_VENV_PYTHON");
    println!("cargo:rustc-check-cfg=cfg(has_torch_hip)");
    println!("cargo:rustc-check-cfg=cfg(has_torch_cuda)");

    if let Some(lib_dir) = libtorch_lib_dir() {
        if lib_dir.join("libtorch_hip.so").exists() {
            println!("cargo:rustc-cfg=has_torch_hip");
        }
        if lib_dir.join("libtorch_cuda.so").exists() {
            println!("cargo:rustc-cfg=has_torch_cuda");
        }
    }
}

use std::fs;
use std::path::Path;

use flate2::read::GzDecoder;

const BASE_URL: &str = "https://systemds.apache.org/assets/datasets/mnist";

const FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

pub fn filenames() -> &'static [&'static str; 4] {
    &FILES
}

pub fn data_dir() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_default();
    dir.push("data");
    dir
}

fn download_and_decompress(url: &str, out_path: &Path) {
    /*
    Download a gzipped file from url and decompress it directly to out_path.
    */

    let response = ureq::get(url)
        .call()
        .expect(&format!("Failed to download {}", url));

    let mut decoder = GzDecoder::new(response.into_reader());
    let mut file = fs::File::create(out_path).expect("Failed to create output file");
    std::io::copy(&mut decoder, &mut file).expect("Failed to decompress");
}

pub fn download_if_needed() {
    let dir = data_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).expect("Failed to create data directory");
    }

    for fname in &FILES {
        let raw_path = dir.join(fname);
        if raw_path.exists() {
            continue;
        }

        let gz_name = format!("{}.gz", fname);
        let url = format!("{}/{}", BASE_URL, gz_name);

        println!("Downloading and decompressing {} ...", gz_name);
        download_and_decompress(&url, &raw_path);
    }
}

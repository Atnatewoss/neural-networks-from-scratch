use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use crate::utils::matrix::Matrix;

#[derive(Serialize, Deserialize)]
struct ModelData {
    n_h: usize,
    W1: Matrix,
    b1: Matrix,
    W2: Matrix,
    b2: Matrix,
}

pub fn save<P: AsRef<Path>>(
    path: P,
    W1: &Matrix,
    b1: &Matrix,
    W2: &Matrix,
    b2: &Matrix,
    n_h: usize,
) {
    let model = ModelData {
        n_h,
        W1: W1.clone(),
        b1: b1.clone(),
        W2: W2.clone(),
        b2: b2.clone(),
    };
    let file = fs::File::create(path).expect("failed to create model file");
    serde_json::to_writer_pretty(file, &model).expect("failed to write model JSON");
    println!("Model saved");
}

pub fn load<P: AsRef<Path>>(path: P) -> (Matrix, Matrix, Matrix, Matrix, usize) {
    let file = fs::File::open(path).expect("failed to open model file");
    let model: ModelData = serde_json::from_reader(file).expect("failed to parse model JSON");
    println!("Model loaded (n_h = {})", model.n_h);
    (model.W1, model.b1, model.W2, model.b2, model.n_h)
}

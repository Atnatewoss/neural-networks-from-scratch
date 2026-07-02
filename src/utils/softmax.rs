use crate::utils::matrix::Matrix;

pub fn forward(Z: &Matrix) -> Matrix {
    let max_per_col = (0..Z.cols)
        .map(|j| {
            let mut max_val = Z.data[j];
            for i in 1..Z.rows {
                let val = Z.data[i * Z.cols + j];
                if val > max_val {
                    max_val = val;
                }
            }
            max_val
        })
        .collect::<Vec<f64>>();

    let mut data = vec![0.0; Z.rows * Z.cols];
    for j in 0..Z.cols {
        let mut col_sum = 0.0;
        for i in 0..Z.rows {
            let ex = (Z.data[i * Z.cols + j] - max_per_col[j]).exp();
            data[i * Z.cols + j] = ex;
            col_sum += ex;
        }
        for i in 0..Z.rows {
            data[i * Z.cols + j] /= col_sum;
        }
    }

    Matrix { data, rows: Z.rows, cols: Z.cols }
}

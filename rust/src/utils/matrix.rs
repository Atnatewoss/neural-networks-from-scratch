use std::fmt;

/*
Matrix: row-major storage
  data[row * cols + col]
*/
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    /*
    Creates a new matrix filled with 0.0
      rows: number of rows
      cols: number of columns
    */
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /*
    Creates a matrix filled with 0.0
      rows: number of rows
      cols: number of columns
    */
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    /*
    Creates a matrix filled with 1.0
      rows: number of rows
      cols: number of columns
    */
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    /*
    Wraps an existing Vec<f64> as a matrix
      data: flat array in row-major order
      rows: number of rows
      cols: number of columns
    */
    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Matrix::from_vec: data length {} != {}x{}",
            data.len(),
            rows,
            cols
        );
        Matrix { data, rows, cols }
    }

    /*
    Returns (rows, cols)
    */
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /*
    Gets the value at row r, column c
    */
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /*
    Sets the value at row r, column c
    */
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    /*
    Reshapes the matrix to new dimensions (same number of elements)
      new_rows: target row count
      new_cols: target column count
    */
    pub fn reshape(&mut self, new_rows: usize, new_cols: usize) {
        assert_eq!(
            self.data.len(),
            new_rows * new_cols,
            "Matrix::reshape: data length {} != {}x{}",
            self.data.len(),
            new_rows,
            new_cols
        );
        self.rows = new_rows;
        self.cols = new_cols;
    }

    /*
    Matrix multiplication: self (rows x k) · other (k x cols) -> (rows x cols)
    */
    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "dot: lhs cols {} != rhs rows {}",
            self.cols, other.rows
        );
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
        result
    }

    /*
    Element-wise addition.
      If other.cols == 1, broadcasts the column vector across self's columns.
    */
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "add: rows mismatch");
        if self.cols == other.cols {
            let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
            Matrix { data, rows: self.rows, cols: self.cols }
        } else if other.cols == 1 {
            let mut result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i * self.cols + j] = self.data[i * self.cols + j] + other.data[i];
                }
            }
            result
        } else {
            panic!("add: shape mismatch ({}x{}) vs ({}x{})", self.rows, self.cols, other.rows, other.cols);
        }
    }

    /*
    Element-wise subtraction (same shape)
    */
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "subtract: rows mismatch");
        assert_eq!(self.cols, other.cols, "subtract: cols mismatch");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Matrix { data, rows: self.rows, cols: self.cols }
    }

    /*
    Element-wise multiplication (same shape)
    */
    pub fn mul_elementwise(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "mul_elementwise: rows mismatch");
        assert_eq!(self.cols, other.cols, "mul_elementwise: cols mismatch");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Matrix { data, rows: self.rows, cols: self.cols }
    }

    /*
    Multiplies every element by a scalar
    */
    pub fn scalar_mul(&self, scalar: f64) -> Matrix {
        let data = self.data.iter().map(|a| a * scalar).collect();
        Matrix { data, rows: self.rows, cols: self.cols }
    }

    /*
    Returns the transpose
    */
    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix { data, rows: self.cols, cols: self.rows }
    }

    /*
    Sums along an axis and returns a row (axis=0) or column (axis=1) vector
    */
    pub fn sum_axis(&self, axis: usize) -> Matrix {
        match axis {
            0 => {
                let mut data = vec![0.0; self.cols];
                for j in 0..self.cols {
                    let mut sum = 0.0;
                    for i in 0..self.rows {
                        sum += self.data[i * self.cols + j];
                    }
                    data[j] = sum;
                }
                Matrix { data, rows: 1, cols: self.cols }
            }
            1 => {
                let mut data = vec![0.0; self.rows];
                for i in 0..self.rows {
                    let mut sum = 0.0;
                    for j in 0..self.cols {
                        sum += self.data[i * self.cols + j];
                    }
                    data[i] = sum;
                }
                Matrix { data, rows: self.rows, cols: 1 }
            }
            _ => panic!("sum_axis: axis must be 0 or 1"),
        }
    }

    /*
    Sums all elements into a single f64
    */
    pub fn sum_all(&self) -> f64 {
        self.data.iter().sum()
    }

    /*
    Applies a function element-wise, returns a new matrix
    */
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        let data = self.data.iter().map(|a| f(*a)).collect();
        Matrix { data, rows: self.rows, cols: self.cols }
    }

    /*
    Applies a function in-place
    */
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for a in self.data.iter_mut() {
            *a = f(*a);
        }
    }

    /*
    Element-wise natural log
    */
    pub fn log(&self) -> Matrix {
        self.map(|x| x.ln())
    }

    /*
    Element-wise exp
    */
    pub fn exp(&self) -> Matrix {
        self.map(|x| x.exp())
    }

    /*
    Element-wise power
    */
    pub fn pow(&self, exp: f64) -> Matrix {
        self.map(|x| x.powf(exp))
    }

    /*
    Prints the shape with a label: "name shape: rows x cols"
    */
    pub fn print_shape(&self, name: &str) {
        println!("{} shape: {}x{}", name, self.rows, self.cols);
    }

    /*
    Returns the index of the maximum value in each column
    */
    pub fn argmax_columns(&self) -> Vec<usize> {
        (0..self.cols)
            .map(|j| {
                let mut max_idx = 0;
                let mut max_val = self.data[j];
                for i in 1..self.rows {
                    let val = self.data[i * self.cols + j];
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                max_idx
            })
            .collect()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows.min(10) {
            write!(f, "[")?;
            for j in 0..self.cols.min(10) {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:8.4}", self.data[i * self.cols + j])?;
            }
            if self.cols > 10 {
                write!(f, ", ...")?;
            }
            writeln!(f, " ]")?;
        }
        if self.rows > 10 {
            writeln!(f, "...")?;
        }
        writeln!(f, "({}x{})", self.rows, self.cols)
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Matrix {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

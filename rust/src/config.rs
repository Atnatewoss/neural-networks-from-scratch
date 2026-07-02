pub struct Config {
    // architecture
    pub n_h: usize,
    // training
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub print_cost: bool,
    // data
    pub val_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_h: 128,
            learning_rate: 0.01,
            num_epochs: 10,
            batch_size: 64,
            print_cost: true,
            val_size: 10000,
        }
    }
}

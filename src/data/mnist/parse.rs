use std::fs;
use std::io::Read;
use std::path::Path;

fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/*
Read raw image bytes from an IDX3 file.
Returns (raw_pixels, num_images, rows, cols).
*/
pub fn read_raw_images(path: &Path) -> (Vec<u8>, usize, usize, usize) {
    /*
    Arguments:
        path -- path to IDX3 image file

    Returns:
        (raw_pixels, num_images, rows, cols)
    */

    let mut file = fs::File::open(path).expect("Failed to open image file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read image file");

    let magic = read_u32_be(&data, 0);
    assert_eq!(magic, 0x00000803, "Invalid image magic number: {:#x}", magic);

    let num_images = read_u32_be(&data, 4) as usize;
    let rows = read_u32_be(&data, 8) as usize;
    let cols = read_u32_be(&data, 12) as usize;

    (data[16..].to_vec(), num_images, rows, cols)
}

/*
Read raw label bytes from an IDX1 file.
Returns (raw_labels, num_labels).
*/
pub fn read_raw_labels(path: &Path) -> (Vec<u8>, usize) {
    /*
    Arguments:
        path -- path to IDX1 label file

    Returns:
        (raw_labels, num_labels)
    */

    let mut file = fs::File::open(path).expect("Failed to open label file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read label file");

    let magic = read_u32_be(&data, 0);
    assert_eq!(magic, 0x00000801, "Invalid label magic number: {:#x}", magic);

    let num_labels = read_u32_be(&data, 4) as usize;

    (data[8..].to_vec(), num_labels)
}

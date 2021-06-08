use ndarray::{ArrayBase, Dimension, ViewRepr};
use serde::Serialize;

use std::fs::File;
use std::io::prelude::*;

use csv::Writer;

// Stores the list of words, separated by new line
// The first line is the length of the list, for preallocation purposes
pub fn write_list(list: &[&str], filename: &str) {
    let mut f = File::create(filename).unwrap();
    for item in list {
        writeln!(f, "{}", item).unwrap();
    }
    let _ = f.flush();
}

pub fn load_list(path: &str) -> Vec<String> {
    let mut f = File::open(path).unwrap();
    let mut unpslit_str = String::new();
    let _ = f.read_to_string(&mut unpslit_str).unwrap();
    unpslit_str.lines().map(String::from).collect()
}

pub fn write_ndarray<T: Dimension>(nd: ArrayBase<ViewRepr<&f64>, T>, path: &str) {
    let mut writer = Writer::from_path(format!("./data/{}.csv", path)).unwrap();
    for record in nd.rows() {
        let slice = record.to_slice().expect("Expected float values");
        let mut byte_slices: Vec<[u8; 8]> = vec![];
        for v in slice {
            byte_slices.push(v.to_be_bytes())
        }
        // let byte_slices: &[[u8; 8]] = slice.iter().map(|v| v.to_le_bytes()).collect_slice();
        let _ = writer.write_record(byte_slices).expect("CSV writer error");
        writer.flush().expect("Flushing writer failed");
    }
}

// pub fn write_csv_vec<T: Serialize>(v: &[Vec<T>], path: &str) {
//     let mut writer = Writer::from_path(path).unwrap();
//     for record in v {
//         let _ = writer.write_record(record).expect("CSV writer error");
//         writer.flush();
//     }
// }

// pub fn write_csv<T: Serialize + Iterator>(nd: &mut T, path: &str) {
//     let mut writer = Writer::from_path(path).unwrap();
//     writer.write_record(nd).expect("CSV writer error");
//     writer.flush();
// }

// pub fn deserialize_from_file<'de, T: serde::Deserialize<'de>>(path: &str) -> T {
pub fn deserialize_from_file(path: &str) -> rustlearn::ensemble::random_forest::RandomForest {
    let mut f = File::open(path).unwrap();
    let mut encoded = Vec::new();

    let _ = f.read_to_end(&mut encoded).unwrap();
    bincode::deserialize(&encoded[..]).unwrap()
}

pub fn serialize_to_file<T>(s: &T, path: &str)
where
    T: Serialize,
{
    let serialized: Vec<u8> = bincode::serialize(s).unwrap();

    let mut f = File::create(path).unwrap();
    let _ = f.write_all(&serialized[..]);
    f.flush().unwrap();
}

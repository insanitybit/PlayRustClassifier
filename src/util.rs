use ndarray::{ArrayBase, Dimension, ViewRepr};

use rustc_serialize::{Encodable, Decodable};

use bincode::SizeLimit;
use bincode::rustc_serialize::{encode, decode};

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
    let mut wtr = Writer::from_file(format!("./data/{}.csv", path)).unwrap();
    // wtr.encode(nd);
    for record in nd.inner_iter() {
        let _ = wtr.encode(record);
    }
}

pub fn write_csv_vec<T: Encodable>(v: &[Vec<T>], path: &str) {
    let mut wtr = Writer::from_file(path).unwrap();
    for record in v {
        let _ = wtr.encode(record);
    }
}

pub fn write_csv<T: Encodable>(nd: &T, path: &str) {
    let mut wtr = Writer::from_file(path).unwrap();
    wtr.encode(nd);
    // for record in nd.inner_iter() {
    //     let _ = wtr.encode(record);
    // }
}

pub fn deserialize_from_file<T: Decodable>(path: &str) -> T {
    let mut f = File::open(path).unwrap();
    let mut encoded = Vec::new();

    let _ = f.read_to_end(&mut encoded).unwrap();
    decode(&encoded[..]).unwrap()
}

pub fn serialize_to_file<T>(s: &T, path: &str)
    where T: Encodable
{
    let serialized: Vec<u8> = encode(&s, SizeLimit::Infinite).unwrap();

    let mut f = File::create(path).unwrap();
    let _ = f.write_all(&serialized[..]);
    f.flush().unwrap();
}

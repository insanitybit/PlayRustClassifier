#![feature(test, custom_derive, plugin)]
#[macro_use(stack)]
extern crate ndarray;
extern crate rustc_serialize;

extern crate rsml;

use rsml::random_forest::RandomForest;
use rustc_serialize::json;

use std::fs::File;
use std::io::prelude::*;

fn load_model() -> RandomForest {
    let mut f = File::open("clf").unwrap();
    let mut json_str = String::new();

    let _ = f.read_to_string(&mut json_str).unwrap();
    json::decode(&json_str).unwrap()
}

fn load_authors() -> Vec<String> {
    let mut f = File::open("total_author_list").unwrap();
    let mut unpslit_str = String::new();
    let _ = f.read_to_string(&mut unpslit_str).unwrap();
    unpslit_str.lines().map(String::from).collect()
}

fn load_all_docs() -> Vec<Vec<(String, usize)>> {
    let mut f = File::open("all_docs").unwrap();
    let mut json_str = String::new();

    let _ = f.read_to_string(&mut json_str).unwrap();
    json::decode(&json_str).unwrap()
}


fn main() {
    let mut rf = load_model();
}

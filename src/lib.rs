#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]

extern crate tiny_keccak;
extern crate serde;
extern crate rayon;
extern crate hyper;
extern crate serde_json;
extern crate stopwatch;
extern crate rustc_serialize;
extern crate csv;
extern crate clap;
extern crate tfidf;
extern crate rsml;

pub mod reddit;
pub mod feature_extraction;

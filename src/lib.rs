#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]

extern crate serde;
extern crate rayon;
extern crate hyper;
extern crate serde_json;
extern crate stopwatch;
extern crate rustc_serialize;
extern crate csv;
extern crate clap;

pub mod reddit;

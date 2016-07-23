#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]

#[macro_use(stack)]
extern crate ndarray;
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

#[macro_export]
macro_rules! time {
    ($expression:expr) => (
        {
            let mut sw = stopwatch::Stopwatch::start_new();
            let exp = $expression;
            sw.stop();
            println!("{} took {},ms",stringify!($expression) , sw.elapsed_ms());
            exp
        }
    );
}

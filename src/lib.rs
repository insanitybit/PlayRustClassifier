#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]


#[macro_export]
macro_rules! time {
    ($expression:expr) => (
        {

            let mut sw = $crate::Stopwatch::start_new();
            let exp = $expression;
            sw.stop();
            println!("{} took {},ms",stringify!($expression) , sw.elapsed_ms());
            exp
        }
    );
    ($expression:expr, $s:expr) => (
        {

            let mut sw = $crate::Stopwatch::start_new();
            let exp = $expression;
            sw.stop();
            println!("{} took {},ms", stringify!($s), sw.elapsed_ms());
            exp
        }
    );
}


#[macro_use(stack)]
extern crate ndarray;
#[macro_use]
extern crate lazy_static;

extern crate clap;
extern crate csv;
extern crate serde;
extern crate hyper;
extern crate rayon;
extern crate regex;
extern crate rsml;
extern crate rustc_serialize;
extern crate serde_json;
extern crate stopwatch;
extern crate tiny_keccak;
extern crate tfidf;

pub mod feature_extraction;
pub mod reddit;
pub mod util;

pub use stopwatch::Stopwatch;

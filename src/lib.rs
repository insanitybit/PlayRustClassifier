
extern crate ndarray;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate serde_derive;

#[macro_export]
macro_rules! time {
    ($expression:expr) => {{
        let mut sw = $crate::Stopwatch::start_new();
        let exp = $expression;
        sw.stop();
        println!("{} took {},ms", stringify!($expression), sw.elapsed_ms());
        exp
    }};
    ($expression:expr, $s:expr) => {{
        let mut sw = $crate::Stopwatch::start_new();
        let exp = $expression;
        sw.stop();
        println!("{} took {},ms", stringify!($s), sw.elapsed_ms());
        exp
    }};
}

pub mod feature_extraction;
pub mod reddit;
pub mod util;

pub use stopwatch::Stopwatch;

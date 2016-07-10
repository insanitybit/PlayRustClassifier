#![feature(test)]

extern crate clap;
extern crate ndarray;
extern crate rayon;
extern crate rsml;
extern crate csv;

use clap::{Arg, App};
use rsml::random_forest::model::*;
use rsml::traits::SupervisedLearning;

fn get_train_csv() -> String {
    let matches = App::new("Model Generator")
                      .version("1.0")
                      .about("Generates a random forest based on a training set")
                      .arg(Arg::with_name("train")
                               .help("The CSV to train on")
                               .required(true)
                               .index(1))
                      .get_matches();

    let train_path = matches.value_of("train").unwrap();


    let mut rdr = csv::Reader::from_file(train_path).unwrap();

    unimplemented!()

}


fn main() {
    let rows = 50000;
    let cols = 10;

    let mut rf = RandomForest::new(10);

    // rf.fit(&train, &answers);

}

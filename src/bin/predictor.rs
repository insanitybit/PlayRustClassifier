#![feature(test, custom_derive, plugin)]
#[macro_use(stack)]
extern crate ndarray;

extern crate clap;
extern crate csv;

extern crate rustc_serialize;
extern crate playrust_alert;

extern crate rsml;

use clap::{Arg, App};
use ndarray::{arr2, Axis, stack, ArrayBase};

use playrust_alert::reddit::{RawPostFeatures, ProcessedPostFeatures};
use playrust_alert::feature_extraction::{convert_author_to_popularity, convert_is_self,
                                         tfidf_reduce_selftext, subs_to_float};

use rsml::random_forest::RandomForest;
use rsml::traits::SupervisedLearning;

use rustc_serialize::json;

use std::fs::File;
use std::io::prelude::*;

fn load_model() -> RandomForest {
    let mut f = File::open("clf").unwrap();
    let mut json_str = String::new();

    let _ = f.read_to_string(&mut json_str).unwrap();
    json::decode(&json_str).unwrap()
}

fn load_unique_word_list() -> Vec<String> {
    let mut f = File::open("unique_word_list").unwrap();
    let mut unpslit_str = String::new();
    let _ = f.read_to_string(&mut unpslit_str).unwrap();
    unpslit_str.lines().map(String::from).collect()
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


fn normalize_post_features(raw_posts: &[RawPostFeatures]) -> (Vec<ProcessedPostFeatures>, Vec<f64>) {
    let selfs: Vec<_> = raw_posts.iter().map(|r| convert_is_self(r.is_self)).collect();
    let downs: Vec<_> = raw_posts.iter().map(|r| r.downs as f64).collect();
    let ups: Vec<_> = raw_posts.iter().map(|r| r.ups as f64).collect();
    let scores: Vec<_> = raw_posts.iter().map(|r| r.score as f64).collect();
    let subreddits: Vec<_> = raw_posts.iter().map(|r| r.subreddit.as_ref()).collect();
    let sub_floats = subs_to_float(&subreddits[..]);
    let mut authors: Vec<String> = raw_posts.into_iter().map(|r| r.author.to_owned()).collect();
    authors.extend_from_slice(&load_authors()[..]);
    let authors: Vec<_> = authors.iter().map(|a| a.as_ref()).collect();
    let posts: Vec<&str> = raw_posts.iter().map(|r| r.selftext.as_ref()).collect();

    let unique_word_list = load_unique_word_list();
    let unique_word_list: Vec<_> = unique_word_list.iter().map(|s| s.as_ref()).collect();

    let all_docs = load_all_docs();
    let tfidf_reduction = tfidf_reduce_selftext(&posts[..], &unique_word_list[..], &all_docs[..]);

    let author_popularity = convert_author_to_popularity(&authors[..]);
    let mut processed = Vec::with_capacity(raw_posts.len());

    for index in 0..raw_posts.len() {
        let p = ProcessedPostFeatures {
            is_self: selfs[index],
            author_popularity: author_popularity[index],
            downs: downs[index],
            ups: ups[index],
            score: scores[index],
            word_freq: tfidf_reduction[index].clone(),
        };
        processed.push(p);
    }
    (processed, sub_floats)
}

fn construct_matrix(post_features: &[ProcessedPostFeatures]) -> ArrayBase<Vec<f64>, (usize, usize)> {
    let auth_pop: Vec<_> = post_features.iter().map(|p| p.author_popularity).collect();
    let downs: Vec<_> = post_features.iter().map(|p| p.downs).collect();
    let ups: Vec<_> = post_features.iter().map(|p| p.ups).collect();
    let score: Vec<_> = post_features.iter().map(|p| p.score).collect();

    assert_eq!(auth_pop.len(), post_features.len());
    assert_eq!(downs.len(), post_features.len());
    assert_eq!(ups.len(), post_features.len());
    assert_eq!(score.len(), post_features.len());

    let term_count = post_features.iter().last().unwrap().word_freq.iter().count();
    let term_frequencies: Vec<_> = post_features.iter().map(|p| &p.word_freq[..]).collect();

    let mut row = vec![auth_pop[0], downs[0], ups[0], score[0]];
    row.extend_from_slice(&term_frequencies[0]);
    let mut a = stack!(Axis(0), row);

    for index in 1..post_features.len() {
        let mut row = vec![auth_pop[index], downs[index], ups[index], score[index]];
        row.extend_from_slice(&term_frequencies[index]);
        a = stack!(Axis(0), a, row);
    }
    a.into_shape((post_features.len(), term_count + 4)).unwrap()
}

fn get_pred_data() -> Vec<RawPostFeatures> {
    let matches = App::new("PlayRust Predictor")
                      .version("1.0")
                      .about("Given a series of reddit posts, predicts which sub they came from")
                      .arg(Arg::with_name("pred")
                               .help("The CSV to predict against")
                               .required(true)
                               .index(1))
                      .get_matches();

    let pred_path = matches.value_of("pred").unwrap();

    let mut rdr = csv::Reader::from_file(pred_path).unwrap();

    rdr.decode()
       .map(|raw_post| raw_post.unwrap())
       .collect()
}

fn main() {
    let raw_posts = get_pred_data();
    let (features, ground_truth) = normalize_post_features(&raw_posts[..]);
    let feat_matrix = construct_matrix(&features[..]);

    let mut rf = load_model();

    rf.predict(&feat_matrix);

}

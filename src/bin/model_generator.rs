#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]
#![feature(test)]

extern crate clap;
extern crate rayon;
extern crate rsml;
extern crate csv;
extern crate rustc_serialize;
extern crate playrust_alert;
extern crate tfidf;
extern crate stopwatch;

use stopwatch::*;

#[macro_use(stack)]
extern crate ndarray;

use ndarray::{arr2, Axis, stack, ArrayBase};

use tfidf::{TfIdf, TfIdfDefault};
use clap::{Arg, App};
use rayon::prelude::*;
use rsml::random_forest::model::*;
use rsml::traits::SupervisedLearning;
use rsml::tfidf_helper::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::*;
use playrust_alert::reddit::RawPostFeatures;

#[derive(Deserialize, Debug, Clone, RustcEncodable)]
pub struct ProcessedPostFeatures {
    /// 0 if self, 1 if not self
    pub is_self: f64,
    /// The popularity of the author relative to the dataset
    pub author_popularity: f64,
    /// The number of downvotes
    pub downs: f64,
    /// The number of upvotes
    pub ups: f64,
    /// The overall score of the post
    pub score: f64,
    /// Whole numbers representing the different subreddits, this is our label
    pub subreddit: f64,
    /// Word frequency vector
    pub word_freq: Vec<f64>,
}

fn get_train_data() -> Vec<RawPostFeatures> {
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

    rdr.decode()
       .map(|raw_post| raw_post.unwrap())
       .collect()
}

fn convert_is_self(b: bool) -> f64 {
    if b {
        0f64
    } else {
        1f64
    }
}

fn convert_author_to_popularity(authors: &[&str]) -> Vec<f64> {
    let mut auth_count = BTreeMap::new();
    for author in authors {
        *auth_count.entry(author).or_insert(0) += 1;
    }
    authors.iter()
           .map(|author| *auth_count.get(author).unwrap() as f64 / authors.len() as f64)
           .collect()
    // auth_count.values().map(|v| *v as f64 / authors.len() as f64).collect()
}

// TODO: This should probably return an ndarray
fn tfidf_reduce_selftext(self_texts: &[&str], words: &[&str]) -> Vec<Vec<f64>> {
    let docs: Vec<_> = {
        let mut docs = Vec::with_capacity(self_texts.len());
        self_texts.par_iter()
                  .map(|s| str_to_doc(s))
                  .collect_into(&mut docs);
        docs
    };
    let docs: Vec<Vec<_>> = docs.iter()
                                .map(|doc| doc.iter().map(|t| (t.0.as_str(), t.1)).collect())
                                .collect();
    let all_docs = docs.clone();

    let mut term_frequency_matrix = Vec::with_capacity(self_texts.len());
    println!("TFIDF over {:?} words and {} docs", words.len(), docs.len());

    for doc in docs.iter() {
        let mut term_frequencies: Vec<f64> = Vec::with_capacity(words.len());

        // let mut sw = stopwatch::Stopwatch::new();
        // sw.start();
        words.par_iter()
             .weight_max()
             .map(|word| TfIdfDefault::tfidf(word, doc, all_docs.iter()))
             .collect_into(&mut term_frequencies);
        // sw.stop();
        // println!("{:?}", sw.elapsed_ms());
        term_frequency_matrix.push(term_frequencies);
    }

    term_frequency_matrix
}

// Stores the list of words, separated by new line
// The first line is the length of the list, for preallocation purposes
fn write_size_and_list(list: &[&str], filename: &str) {
    let mut f = File::create(filename).unwrap();
    writeln!(f, "{}", list.len());
    for item in list {
        writeln!(f, "{}", item);
    }
    f.flush().unwrap();
}

fn subs_to_float(subs: &[&str]) -> Vec<f64> {
    let mut sub_float_map = BTreeMap::new();
    let mut sub_floats = Vec::with_capacity(subs.len());
    let mut cur_sub = 0f64;

    for sub in subs {
        let f = *sub_float_map.entry(sub).or_insert({
            let c = cur_sub;
            cur_sub += 1f64;
            c
        });
        sub_floats.push(f);
    }
    sub_floats
}

fn normalize_post_features(raw_posts: &[RawPostFeatures]) -> Vec<ProcessedPostFeatures> {
    let selfs: Vec<_> = raw_posts.iter().map(|r| convert_is_self(r.is_self)).collect();
    let downs: Vec<_> = raw_posts.iter().map(|r| r.downs as f64).collect();
    let ups: Vec<_> = raw_posts.iter().map(|r| r.ups as f64).collect();
    let scores: Vec<_> = raw_posts.iter().map(|r| r.score as f64).collect();
    let mut authors: Vec<&str> = raw_posts.iter().map(|r| r.author.as_ref()).collect();
    let posts: Vec<&str> = raw_posts.iter().map(|r| r.selftext.as_ref()).collect();
    let subreddits: Vec<_> = raw_posts.iter().map(|r| r.subreddit.as_ref()).collect();
    let sub_floats = subs_to_float(&subreddits[..]);

    let unique_word_list = get_unique_word_list(&posts[..]);
    let unique_word_list: Vec<_> = unique_word_list.iter().map(|s| s.as_ref()).collect();
    let tfidf_reduction = tfidf_reduce_selftext(&posts[..], &unique_word_list[..]);
    let author_popularity = convert_author_to_popularity(&authors[..]);

    authors.sort();
    authors.dedup();
    write_size_and_list(&authors[..], "./unique_author_list");
    write_size_and_list(&unique_word_list[..], "./unique_word_list");

    let mut processed = Vec::with_capacity(tfidf_reduction.len());

    for index in 0..tfidf_reduction.len() {
        let p = ProcessedPostFeatures {
            is_self: selfs[index],
            author_popularity: author_popularity[index],
            downs: downs[index],
            ups: ups[index],
            score: scores[index],
            subreddit: sub_floats[index],
            word_freq: tfidf_reduction[index].clone(),
        };
        processed.push(p);
    }
    processed
}

fn construct_matrix(post_features: &[ProcessedPostFeatures]) -> ArrayBase<Vec<f64>, (usize, usize)> {
    let selfs: Vec<_> = post_features.iter().map(|p| p.is_self).collect();
    let auth_pop: Vec<_> = post_features.iter().map(|p| p.author_popularity).collect();
    let downs: Vec<_> = post_features.iter().map(|p| p.downs).collect();
    let ups: Vec<_> = post_features.iter().map(|p| p.ups).collect();
    let score: Vec<_> = post_features.iter().map(|p| p.score).collect();

    assert_eq!(selfs.len(), post_features.len());
    assert_eq!(auth_pop.len(), post_features.len());
    assert_eq!(downs.len(), post_features.len());
    assert_eq!(ups.len(), post_features.len());
    assert_eq!(score.len(), post_features.len());
    let mut a = stack!(Axis(0), selfs, auth_pop, downs, ups, score);

    let term_count = post_features.iter().last().unwrap().word_freq.iter().count();
    for term_frequency in post_features.iter().map(|p| &p.word_freq[..]) {
        a = stack!(Axis(0), a, term_frequency);
    }

    a.into_shape((post_features.len(), term_count + 5)).unwrap()
}

fn main() {
    let posts = get_train_data();
    let features = normalize_post_features(&posts[..]);
    let feat_matrix = construct_matrix(&features[..]);
    let ground_truth: Vec<_> = features.iter().map(|p| p.subreddit).collect();

    println!("building the random forest");
    let mut rf = RandomForest::new(10);
    rf.fit(&feat_matrix, &stack!(Axis(0), ground_truth));

    // println!("{:?}", rf.predict(&feat_matrix));
}

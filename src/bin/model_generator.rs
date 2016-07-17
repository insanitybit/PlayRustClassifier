#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]
#![feature(test)]

extern crate clap;
extern crate ndarray;
extern crate rayon;
extern crate rsml;
extern crate csv;
extern crate rustc_serialize;
extern crate playrust_alert;
extern crate tfidf;

extern crate stopwatch;
use stopwatch::Stopwatch;

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

    // let mut term_frequency_cache = BTreeMap::new();

    let mut sw = Stopwatch::new();
    for doc in docs.iter() {
        sw.start();
        let mut term_frequencies: Vec<_> = Vec::with_capacity(words.len());
        term_frequencies = words.iter()
                                .map(|word| {
                                    TfIdfDefault::tfidf(word, doc, all_docs.iter())
                                    // *term_frequency_cache.entry(word)
                                    //   .or_insert(TfIdfDefault::tfidf(word, doc, all_docs.iter()))
                                })
                                .collect();

        term_frequency_matrix.push(term_frequencies);

        sw.stop();
        println!("{:?}ms", sw.elapsed_ms());
        sw.reset();
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

fn normalize_post_features(raw_posts: &[RawPostFeatures]) -> Vec<ProcessedPostFeatures> {

    let selfs: Vec<_> = raw_posts.iter().map(|r| convert_is_self(r.is_self)).collect();
    println!("selfs");
    let downs: Vec<_> = raw_posts.iter().map(|r| r.downs as f64).collect();
    println!("downs");
    let ups: Vec<_> = raw_posts.iter().map(|r| r.ups as f64).collect();
    println!("ups");
    let score: Vec<_> = raw_posts.iter().map(|r| r.score as f64).collect();
    println!("score");
    let mut authors: Vec<&str> = raw_posts.iter().map(|r| r.author.as_ref()).collect();
    println!("authors");
    let posts: Vec<&str> = raw_posts.iter().map(|r| r.selftext.as_ref()).collect();
    println!("posts");

    let author_popularity = convert_author_to_popularity(&authors[..]);
    println!("popularity");

    let unique_word_list = get_unique_word_list(&posts[..]);
    let unique_word_list: Vec<_> = unique_word_list.iter().map(|s| s.as_ref()).collect();
    write_size_and_list(&unique_word_list[..], "./unique_word_list");
    let tfidf_reduction = tfidf_reduce_selftext(&posts[..], &unique_word_list[..]);

    assert_eq!(selfs.len(), raw_posts.len(), "selfs");
    assert_eq!(downs.len(), raw_posts.len(), "downs");
    assert_eq!(ups.len(), raw_posts.len(), "ups");
    assert_eq!(score.len(), raw_posts.len(), "score");
    assert_eq!(authors.len(), raw_posts.len(), "authors");
    assert_eq!(author_popularity.len(),
               raw_posts.len(),
               "author_popularity");
    assert_eq!(posts.len(), raw_posts.len(), "posts");
    assert_eq!(tfidf_reduction.len(), raw_posts.len(), "tfidf_reduction");

    write_size_and_list(&unique_word_list[..], "./unique_word_list");
    authors.sort();
    authors.dedup();
    write_size_and_list(&authors[..], "./unique_author_list");

    vec![]
}


fn main() {
    let posts = get_train_data();
    println!("Get training data");
    normalize_post_features(&posts[..]);

    // let mut rf = RandomForest::new(10);

    // rf.fit(&train, &answers);

}

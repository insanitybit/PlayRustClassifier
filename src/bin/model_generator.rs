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

use tfidf::{TfIdf, TfIdfDefault};
use clap::{Arg, App};
use rsml::random_forest::model::*;
use rsml::traits::SupervisedLearning;
use rsml::tfidf_helper::*;
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

fn convert_is_self(b: bool) -> f64 {
    if b {
        0f64
    } else {
        1f64
    }
}

fn convert_author_to_popularity(authors: &[&str]) -> Vec<f64> {
    unimplemented!()
}

// Generate the full list of words and export it.
// TODO: This should probably return an ndarray
fn tfidf_reduce_selftext(self_texts: &[&str], words: &[&str]) -> Vec<Vec<f64>> {
    let docs: Vec<_> = self_texts.iter().map(|s| str_to_doc(s)).collect();
    let all_docs = docs.clone();
    let mut term_frequency_matrix = Vec::with_capacity(self_texts.len());

    for doc in docs.into_iter() {
        let term_frequencies: Vec<_> = words.iter()
                                            .map(|word| {
                                                TfIdfDefault::tfidf(word, &doc, all_docs.iter())
                                            })
                                            .collect();
        term_frequency_matrix.push(term_frequencies);
    }

    term_frequency_matrix
}

fn normalize_post_features(raw_posts: &[RawPostFeatures]) -> Vec<ProcessedPostFeatures> {

    let selfs: Vec<_> = raw_posts.iter().map(|r| convert_is_self(r.is_self)).collect();
    let downs: Vec<_> = raw_posts.iter().map(|r| r.downs as f64).collect();
    let ups: Vec<_> = raw_posts.iter().map(|r| r.ups as f64).collect();
    let score: Vec<_> = raw_posts.iter().map(|r| r.score as f64).collect();
    let authors: Vec<&str> = raw_posts.iter().map(|r| r.author.as_ref()).collect();
    let posts: Vec<&str> = raw_posts.iter().map(|r| r.selftext.as_ref()).collect();

    let unique_word_list = get_unique_word_list(&posts[..]);
    // TODO: Write the unique_word_list out to a file
    let tfidf_reduction = tfidf_reduce_selftext(&posts[..], &unique_word_list[..]);


    let author_popularity = convert_author_to_popularity(&authors[..]);
    assert_eq!(selfs.len(), raw_posts.len());
    assert_eq!(downs.len(), raw_posts.len());
    assert_eq!(ups.len(), raw_posts.len());
    assert_eq!(score.len(), raw_posts.len());
    assert_eq!(authors.len(), raw_posts.len());
    assert_eq!(author_popularity.len(), raw_posts.len());
    assert_eq!(posts.len(), raw_posts.len());
    assert_eq!(tfidf_reduction.len(), raw_posts.len());

    unimplemented!()
}


fn main() {
    let rows = 50000;
    let cols = 10;

    let mut rf = RandomForest::new(10);

    // rf.fit(&train, &answers);

}

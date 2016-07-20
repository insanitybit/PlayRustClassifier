#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]
#![feature(test)]

extern crate clap;
extern crate dedup_by;
extern crate rayon;
extern crate rand;
extern crate rsml;
extern crate csv;
extern crate rustc_serialize;
extern crate serde_json;
extern crate playrust_alert;
extern crate tfidf;
extern crate stopwatch;

use stopwatch::*;
use dedup_by::dedup_by;
use std::mem;
use std::cmp::Ordering;
use rustc_serialize::json;
#[macro_use(stack)]
extern crate ndarray;

use ndarray::{arr2, Axis, stack, ArrayBase};

use tfidf::{TfIdf, TfIdfDefault};
use clap::{Arg, App};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rsml::random_forest::model::*;
use rsml::traits::SupervisedLearning;
use rsml::tfidf_helper::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::prelude::*;
use playrust_alert::reddit::RawPostFeatures;

#[derive(Debug, Clone, RustcEncodable)]
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
fn tfidf_reduce_selftext(self_texts: &[&str],
                         words: &[&str])
                         -> (Vec<Vec<f64>>, Vec<Vec<(String, usize)>>) {
    let docs: Vec<_> = {
        let mut docs = Vec::with_capacity(self_texts.len());
        self_texts.par_iter()
                  .map(|s| str_to_doc(s))
                  .collect_into(&mut docs);
        docs
    };
    let owned_docs = docs.clone();
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

    (term_frequency_matrix, owned_docs)
}

// Stores the list of words, separated by new line
// The first line is the length of the list, for preallocation purposes
fn write_size_and_list(list: &[&str], filename: &str) {
    let mut f = File::create(filename).unwrap();
    for item in list {
        writeln!(f, "{}", item);
    }
    f.flush().unwrap();
}

fn subs_to_float(subs: &[&str]) -> Vec<f64> {
    // let mut sub_float_map = BTreeMap::new();
    let mut sub_floats = Vec::with_capacity(subs.len());
    let mut cur_sub = 0f64;

    for sub in subs {
        if *sub == "rust" {
            sub_floats.push(0f64)
        } else if *sub == "playrust" {
            sub_floats.push(1f64)
        } else {
            panic!("{}", sub);
        }
        // let f = *sub_float_map.entry(sub).or_insert({
        //     let c = cur_sub;
        //     cur_sub += 1f64;
        //     c
        // });
        // sub_floats.push(f);
    }
    println!("{:?}", sub_floats);
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
    let (tfidf_reduction, all_docs) = tfidf_reduce_selftext(&posts[..], &unique_word_list[..]);
    let author_popularity = convert_author_to_popularity(&authors[..]);

    authors.sort();
    write_size_and_list(&authors[..], "./total_author_list");
    write_size_and_list(&unique_word_list[..], "./unique_word_list");
    save_all_docs(&all_docs[..]);

    let mut processed = Vec::with_capacity(raw_posts.len());

    for index in 0..raw_posts.len() {
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

fn write_features(nd: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, (usize, usize)>, path: &str) {
    let mut wtr = csv::Writer::from_file(format!("./{}.csv", path)).unwrap();
    // wtr.encode(nd);
    for record in nd.inner_iter() {
        let _ = wtr.encode(record);
    }
}

fn write_truth(nd: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, usize>, path: &str) {
    let mut wtr = csv::Writer::from_file(format!("./{}.csv", path)).unwrap();
    // wtr.encode(nd);
    for record in nd.inner_iter() {
        let _ = wtr.encode(record);
    }
}

fn save_all_docs(docs: &[Vec<(String, usize)>]) {
    let serialized = json::encode(&docs).unwrap();

    let mut f = File::create("all_docs").unwrap();
    write!(f, "{}", serialized);
    f.flush().unwrap();
}

fn save_rf(rf: &RandomForest) {
    let serialized = json::encode(&rf).unwrap();

    let mut f = File::create("clf").unwrap();
    write!(f, "{}", serialized);
    f.flush().unwrap();
}

fn main() {
    let posts = get_train_data();
    println!("{:?}", posts.len());
    let mut posts: Vec<_> = posts.into_iter().filter(|post| post.is_self).collect();
    let init_s = posts.len();
    posts.sort_by(|a, b| a.title.cmp(&b.title));
    dedup_by(&mut posts, |a, b| a.title == b.title);
    println!("{}, {:?}", init_s, posts.len());
    let mut rng = thread_rng();

    rng.shuffle(&mut posts);

    let features = normalize_post_features(&posts[..]);
    let feat_matrix = construct_matrix(&features[..]);
    let ground_truth: Vec<_> = features.iter().map(|p| p.subreddit).collect();
    let ground_truth = &stack!(Axis(0), ground_truth);
    let (truth1, truth2) = ground_truth.view().split_at(Axis(0), posts.len() / 6);
    let (sample1, sample2) = feat_matrix.view().split_at(Axis(0), posts.len() / 6);

    write_truth(truth1, "truth1");
    write_truth(truth2, "truth2");
    write_features(sample1, "sample1");
    write_features(sample2, "sample2");

    let (pred_raw, _) = posts.split_at(posts.len() / 6);
    println!("{:?} {:?}", truth1.shape(), truth2.shape());
    println!("building the random forest");

    let mut rf = RandomForest::new(20);
    rf.fit(&sample2.to_owned(), &truth2.to_owned());

    save_rf(&rf);

    let preds = rf.predict(&sample1.to_owned()).unwrap();

    let mut hits = 0;
    let mut miss = 0;
    for (index, (pred, truth)) in preds.iter().zip(truth1.iter()).enumerate() {
        let normal_pred = {
            if *pred > 0.6 {
                1f64
            } else {
                0f64
            }
        };

        let truth = truth.round();
        println!("{:?} {:?} {:?}", pred, normal_pred, truth);

        if normal_pred == truth {
            hits += 1;
        } else {
            miss += 1;
            // println!("{:?}", pred_raw[index]);
        }
    }
    println!("hit: {}\nmiss: {}", hits, miss);
    // println!("{:?}", unique_subs);
}

#[cfg(test)]
mod tests {
    use super::{subs_to_float, dedup_by};

    // #[test]
    // fn test_subs_to_float() {
    //     let subs = vec!["a", "b", "c", "c", "b"];
    //     assert_eq!(vec![0f64, 1f64, 2f64, 2f64, 1f64], subs_to_float(&subs[..]))
    // }
}

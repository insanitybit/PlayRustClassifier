extern crate clap;
extern crate csv;
extern crate dedup_by;
extern crate rand;
extern crate rayon;
extern crate rsml;
// extern crate rustc_serialize;
extern crate serde_json;
extern crate stopwatch;
extern crate tfidf;

use clap::{App, Arg};
use dedup_by::dedup_by;
use playrust_alert::reddit::RawPostFeatures;
// use rsml::tfidf_helper::get_unique_word_list;

use std::collections::BTreeMap;

fn get_train_data() -> Vec<RawPostFeatures> {
    let matches = App::new("Model Generator")
        .version("1.0")
        .about("Generates a random forest based on a training set")
        .arg(
            Arg::with_name("train")
                .help("The CSV to train on")
                .required(true)
                .index(1),
        )
        .get_matches();

    let train_path = matches.value_of("train").unwrap();

    let rdr = csv::Reader::from_path(train_path).unwrap();

    let mut posts: Vec<RawPostFeatures> = rdr.into_deserialize().map(|v| v.expect("Failed to deserialize train data for generate freqs")).collect();

    posts.sort_by(|a, b| a.title.cmp(&b.title));
    dedup_by(&mut posts, |a, b| a.title == b.title);
    posts
}

// normalize: Option<F>, 
/// Replacing the missing code from rsml
fn get_unique_words<'a>(words: &'a [&str]) -> std::collections::HashSet<&'a str> {
    let mut set = std::collections::HashSet::new();
    for word in words {
        set.insert(*word);
    }
    set
}

fn word_freqs(posts: &[RawPostFeatures]) -> BTreeMap<String, u64> {
    let mut map = BTreeMap::new();

    for post in posts {
        let post = vec![post.selftext.as_str()];

        for word in get_unique_words(&post[..]) {
            *map.entry(word.to_string()).or_insert(0) += 1;
        }
    }
    map
}

fn main() {
    let posts = get_train_data();
    let (rust, _play): (Vec<RawPostFeatures>, Vec<RawPostFeatures>) =
        posts.into_iter().partition(|post| post.subreddit == "rust");

    let mut rust_word_freq: Vec<(String, u64)> = word_freqs(&rust[..]).into_iter().collect();
    rust_word_freq.sort_by(|a, b| a.1.cmp(&b.1));

    // let mut play_word_freq: Vec<(String, u64)> = word_freqs(&play[..]).into_iter().collect();
    // play_word_freq.sort_by(|a, b| a.1.cmp(&b.1));

    println!("{:#?}", rust_word_freq);
}

#![feature(custom_derive, plugin)]
#![plugin(serde_macros)]
#![feature(test)]

#[macro_use(time)]
extern crate playrust_alert;

#[macro_use(stack)]
extern crate ndarray;

extern crate clap;
extern crate csv;
extern crate dedup_by;
extern crate rand;
extern crate rayon;
extern crate rsml;
extern crate rustc_serialize;
extern crate serde_json;
extern crate stopwatch;

use clap::{Arg, App};
use dedup_by::dedup_by;
use ndarray::{Axis, ArrayBase, Array};

use playrust_alert::reddit::{RawPostFeatures, ProcessedPostFeatures};
use playrust_alert::feature_extraction::{convert_author_to_popularity, convert_is_self,
                                         check_for_code, subs_to_float, interesting_word_freq,
                                         symbol_counts};

use playrust_alert::util::*;

use rand::{thread_rng, Rng};
use rsml::random_forest::model::*;
use rsml::traits::SupervisedLearning;

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

    let posts: Vec<RawPostFeatures> = rdr.decode()
                                         .map(|raw_post| raw_post.unwrap())
                                         .collect();

    let mut posts: Vec<RawPostFeatures> = posts.into_iter()
                                               .filter(|raw_post| raw_post.selftext.len() > 8)
                                               .collect();
    println!("{:?}", posts.len());
    posts.sort_by(|a, b| a.title.cmp(&b.title));
    dedup_by(&mut posts, |a, b| a.title == b.title);
    posts
}


fn normalize_post_features(raw_posts: &[RawPostFeatures]) -> (Vec<ProcessedPostFeatures>, Vec<f64>) {
    let selfs: Vec<_> = raw_posts.iter().map(|r| convert_is_self(r.is_self)).collect();
    let downs: Vec<_> = raw_posts.iter().map(|r| r.downs as f64).collect();
    let ups: Vec<_> = raw_posts.iter().map(|r| r.ups as f64).collect();
    let scores: Vec<_> = raw_posts.iter().map(|r| r.score as f64).collect();
    let mut authors: Vec<&str> = raw_posts.iter().map(|r| r.author.as_ref()).collect();
    let rust_authors: Vec<&str> = raw_posts.iter()
                                           .filter_map(|r| if r.subreddit == "rust" {
                                               Some(r.author.as_ref())
                                           } else {
                                               None
                                           })
                                           .collect();
    let posts: Vec<&str> = raw_posts.iter().map(|r| r.selftext.as_ref()).collect();
    let post_lens: Vec<f64> = raw_posts.iter().map(|r| r.selftext.len() as f64).collect();

    let titles: Vec<&str> = raw_posts.iter().map(|r| r.title.as_ref()).collect();
    let subreddits: Vec<_> = raw_posts.iter().map(|r| r.subreddit.as_ref()).collect();
    let sub_floats = subs_to_float(&subreddits[..]);

    let interesting_words = load_list("./static_data/words_of_interest");

    let mut terms = Vec::new();

    for (post, title) in posts.iter().zip(titles.iter()) {
        let mut comb = String::new();
        comb.push_str(post);
        comb.push_str(" ");
        comb.push_str(title);
        terms.push(comb);
    }

    let terms: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

    let term_frequencies = interesting_word_freq(&terms[..], &interesting_words[..]);
    let symbol_frequences = symbol_counts(&posts[..]);
    let rust_regexes = check_for_code(&posts[..]);

    let author_popularity = convert_author_to_popularity(&authors[..], &rust_authors[..]);

    authors.sort();
    write_list(&rust_authors[..], "./data/rust_author_list");

    let mut processed = Vec::with_capacity(raw_posts.len());

    for index in 0..raw_posts.len() {
        let p = ProcessedPostFeatures {
            is_self: selfs[index],
            author_popularity: author_popularity[index],
            downs: downs[index],
            ups: ups[index],
            score: scores[index],
            word_freq: term_frequencies[index].clone(),
            symbol_freq: symbol_frequences[index].clone(),
            post_len: post_lens[index],
            regex_matches: rust_regexes[index].clone(),
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
    let post_lens: Vec<_> = post_features.iter().map(|p| p.post_len).collect();

    let term_count = post_features.iter().last().unwrap().word_freq.iter().count();
    let term_count = term_count + post_features.iter().last().unwrap().symbol_freq.iter().count();
    let term_count = term_count + post_features.iter().last().unwrap().regex_matches.iter().count();

    let term_frequencies: Vec<_> = post_features.iter().map(|p| &p.word_freq[..]).collect();
    let symbol_frequencies: Vec<_> = post_features.iter().map(|p| &p.symbol_freq[..]).collect();
    let regex_matches: Vec<_> = post_features.iter().map(|p| &p.regex_matches[..]).collect();

    let mut row = vec![auth_pop[0], downs[0], ups[0], score[0], post_lens[0]];
    let term_count = term_count + row.len();

    row.extend_from_slice(term_frequencies[0]);
    row.extend_from_slice(symbol_frequencies[0]);
    row.extend_from_slice(regex_matches[0]);
    let mut a = stack!(Axis(0), row);

    for index in 1..post_features.len() {
        let mut row = vec![auth_pop[index],
                           downs[index],
                           ups[index],
                           score[index],
                           post_lens[index]];
        row.extend_from_slice(term_frequencies[index]);
        row.extend_from_slice(symbol_frequencies[index]);
        row.extend_from_slice(regex_matches[index]);
        a = stack!(Axis(0), a, row);
    }
    a.into_shape((post_features.len(), term_count)).unwrap()
}

fn main() {
    // Deserialize raw reddit post features from an input file, deduplicate by the title, and
    // then shuffle them.

    let posts: Vec<_> = {
        let mut posts = get_train_data();
        let mut rng = thread_rng();
        rng.shuffle(&mut posts);
        // posts.into_iter().take(500).collect()
        posts
    };

    // Generate our processed feature matrix
    let (features, ground_truth) = normalize_post_features(&posts[..]);
    let feat_matrix = construct_matrix(&features[..]);

    // Split our data such that we train on one set and can test our accuracy on another
    let ground_truth = Array::from_vec(ground_truth);

    let (truth1, truth2) = ground_truth.view().split_at(Axis(0), posts.len() / 9);
    let (sample1, sample2) = feat_matrix.view().split_at(Axis(0), posts.len() / 9);
    write_ndarray(truth1, "truth1");
    write_ndarray(truth2, "truth2");
    write_ndarray(sample1, "sample1");
    write_ndarray(sample2, "sample2");

    let mut rf = RandomForest::new(1);
    rf.fit(&sample2.to_owned(), &truth2.to_owned());

    serialize_to_file(&rf, "./models/rf");

    let preds = rf.predict(&sample1.to_owned()).unwrap();

    let mut hits = 0;
    let mut miss = 0;
    for (pred, truth) in preds.iter().zip(truth1.iter()) {
        let normal_pred = {
            if *pred > 0.6 {
                1f64
            } else {
                0f64
            }
        };

        let truth = truth.round();
        // println!("{:?} {:?} {:?}", pred, normal_pred, truth);

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
    use super::*;
    use super::playrust_alert::feature_extraction::subs_to_float;
    #[test]
    fn test_subs_to_float() {
        let subs = vec!["a", "b", "c", "c", "b", "d", "a"];
        assert_eq!(vec![0f64, 1f64, 2f64, 2f64, 1f64, 3f64, 0f64],
                   subs_to_float(&subs[..]))
    }
}

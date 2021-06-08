#[macro_use(time)]
extern crate playrust_alert;

extern crate clap;
extern crate csv;
extern crate dedup_by;
extern crate rand;
extern crate rayon;
extern crate rustlearn;
extern crate serde_json;
extern crate stopwatch;

use clap::{App, Arg};

use rustlearn::cross_validation::cross_validation::CrossValidation;

use dedup_by::dedup_by;

use playrust_alert::feature_extraction::{
    check_for_code, convert_author_to_popularity, convert_is_self, interesting_word_freq,
    subs_to_float, symbol_counts,
};
use playrust_alert::reddit::{ProcessedPostFeatures, RawPostFeatures};

use playrust_alert::util::*;

use rustlearn::ensemble::random_forest::Hyperparameters;
use rustlearn::metrics::accuracy_score;
use rustlearn::prelude::*;
use rustlearn::trees::decision_tree;

use rand::{thread_rng, StdRng};

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

    let posts: Vec<RawPostFeatures> = rdr.into_deserialize().map(|v| v.expect("Failed to deserialize train data")).collect();

    let mut posts: Vec<RawPostFeatures> = posts
        .into_iter()
        .filter(|raw_post| raw_post.selftext.len() > 8)
        .collect();

    posts.sort_by(|a, b| a.title.cmp(&b.title));
    dedup_by(&mut posts, |a, b| a.title == b.title);
    posts
}

fn extract_post_features(raw_posts: &[RawPostFeatures]) -> (Vec<ProcessedPostFeatures>, Array) {
    let selfs: Vec<_> = raw_posts
        .iter()
        .map(|r| convert_is_self(r.is_self))
        .collect();
    let downs: Vec<_> = raw_posts.iter().map(|r| r.downs as f32).collect();
    let ups: Vec<_> = raw_posts.iter().map(|r| r.ups as f32).collect();
    let scores: Vec<_> = raw_posts.iter().map(|r| r.score as f32).collect();
    let mut authors: Vec<&str> = raw_posts.iter().map(|r| r.author.as_ref()).collect();

    write_list(&authors[..], "./data/all_authors");
    let rust_authors: Vec<&str> = raw_posts
        .iter()
        .filter_map(|r| {
            if r.subreddit == "rust" {
                Some(r.author.as_ref())
            } else {
                None
            }
        })
        .collect();
    let posts: Vec<&str> = raw_posts.iter().map(|r| r.selftext.as_ref()).collect();
    let post_lens: Vec<f32> = raw_posts.iter().map(|r| r.selftext.len() as f32).collect();

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

    let term_frequencies = time!(interesting_word_freq(&terms[..], &interesting_words[..]));
    let symbol_frequences = time!(symbol_counts(&posts[..]));
    let rust_regexes = time!(check_for_code(&posts[..]));

    let author_popularity = time!(convert_author_to_popularity(
        &authors[..],
        &rust_authors[..]
    ));

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

    (processed, Array::from(sub_floats))
}

fn construct_matrix(post_features: &[ProcessedPostFeatures]) -> Array {
    let auth_pop: Vec<_> = post_features.iter().map(|p| p.author_popularity).collect();
    let downs: Vec<_> = post_features.iter().map(|p| p.downs).collect();
    let ups: Vec<_> = post_features.iter().map(|p| p.ups).collect();
    let score: Vec<_> = post_features.iter().map(|p| p.score).collect();
    let post_lens: Vec<_> = post_features.iter().map(|p| p.post_len).collect();

    let feature_count = post_features
        .iter()
        .last()
        .unwrap()
        .word_freq
        .iter()
        .count();
    let feature_count = feature_count
        + post_features
            .iter()
            .last()
            .unwrap()
            .symbol_freq
            .iter()
            .count();
    let feature_count = feature_count
        + post_features
            .iter()
            .last()
            .unwrap()
            .regex_matches
            .iter()
            .count();

    let term_frequencies: Vec<_> = post_features.iter().map(|p| &p.word_freq[..]).collect();
    let symbol_frequencies: Vec<_> = post_features.iter().map(|p| &p.symbol_freq[..]).collect();
    let regex_matches: Vec<_> = post_features.iter().map(|p| &p.regex_matches[..]).collect();

    let feature_count = feature_count + 5;

    let mut features = Vec::with_capacity(feature_count * post_features.len());

    for index in 0..post_features.len() {
        let row = vec![
            auth_pop[index],
            downs[index],
            ups[index],
            score[index],
            post_lens[index],
        ];
        features.extend_from_slice(&row[..]);
        features.extend_from_slice(term_frequencies[index]);
        features.extend_from_slice(symbol_frequencies[index]);
        features.extend_from_slice(regex_matches[index]);
    }

    let mut features = Array::from(features);
    features.reshape(post_features.len(), feature_count);
    features
}

fn main() {
    use rand::{Rng, SeedableRng};
    // Deserialize raw reddit post features from an input file, deduplicate by the title, and
    // then shuffle them.
    let posts: Vec<_> = {
        let mut posts = get_train_data();
        let mut rng = thread_rng();
        rng.shuffle(&mut posts);
        // posts.into_iter().take(10).collect()
        posts
    };

    let (features, ground_truth) = extract_post_features(&posts[..]);
    let feat_matrix: Array = construct_matrix(&features[..]);

    let tree_params = decision_tree::Hyperparameters::new(feat_matrix.cols());

    let mut model = Hyperparameters::new(tree_params, 10)
        .rng(StdRng::from_seed(&[100]))
        .one_vs_rest();

    model.fit_parallel(&feat_matrix, &ground_truth, 8).unwrap();
    serialize_to_file(&model, "./models/rustlearnrf");

    let no_splits = 10;

    let mut cv = CrossValidation::new(feat_matrix.rows(), no_splits);
    cv.set_rng(StdRng::from_seed(&[100]));

    let mut test_accuracy = 0.0;
    for (train_idx, test_idx) in cv {
        let x_train = feat_matrix.get_rows(&train_idx);
        let x_test = feat_matrix.get_rows(&test_idx);

        let y_train = ground_truth.get_rows(&train_idx);
        model.fit_parallel(&x_train, &y_train, 8).unwrap();
        let test_prediction = model.predict(&x_test).unwrap();

        // println!("test_prediction {:#?}", test_prediction);
        test_accuracy += accuracy_score(&ground_truth.get_rows(&test_idx), &test_prediction);
    }
    test_accuracy /= no_splits as f32;

    println!("Accuracy {}", test_accuracy);
}

#[cfg(test)]
mod tests {
    use super::playrust_alert::feature_extraction::subs_to_float;

    #[test]
    fn test_subs_to_float() {
        let subs = vec!["a", "b", "c", "c", "b", "d", "a"];
        assert_eq!(
            vec![0f32, 1f32, 2f32, 2f32, 1f32, 3f32, 0f32],
            subs_to_float(&subs[..])
        )
    }
}

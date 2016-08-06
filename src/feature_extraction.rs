use rayon::prelude::*;
use regex::Regex;
use rsml::tfidf_helper::*;
use tfidf::{TfIdf, TfIdfDefault};
use std::collections::BTreeMap;

pub fn convert_is_self(b: bool) -> f64 {
    if b {
        0f64
    } else {
        1f64
    }
}

pub fn convert_author_to_popularity<T: AsRef<str>>(authors: &[T],
                                                   rust_authors: &[&str])
                                                   -> Vec<f64> {
    let mut auth_count: BTreeMap<&str, _> = BTreeMap::new();

    for author in authors {
        if rust_authors.contains(&author.as_ref()) {
            *auth_count.entry(author.as_ref()).or_insert(0) += 1;
        }
    }
    let mut freqs = Vec::with_capacity(authors.len());

    for author in authors {
        let author_freq = *auth_count.get(author.as_ref()).unwrap_or(&0);
        freqs.push(author_freq as f64);
    }
    freqs
}

pub fn text_to_docs<T: AsRef<str>>(texts: &[&str]) -> Vec<Vec<(String, usize)>> {
    let mut docs = Vec::with_capacity(texts.len());
    texts.par_iter()
         .map(|s| str_to_doc(s))
         .collect_into(&mut docs);
    docs
}

// TODO: This should probably return an ndarray
pub fn tfidf_reduce_selftext(self_texts: &[&str],
                             words: &[&str],
                             docs: &[Vec<(String, usize)>])
                             -> Vec<Vec<f64>> {

    let docs: Vec<Vec<_>> = docs.iter()
                                .map(|doc| doc.iter().map(|t| (t.0.as_str(), t.1)).collect())
                                .collect();
    let all_docs = docs.clone();

    let mut term_frequency_matrix = Vec::with_capacity(self_texts.len());
    println!("TFIDF over {:?} words and {} docs", words.len(), docs.len());

    for doc in docs {
        let mut term_frequencies: Vec<f64> = Vec::with_capacity(words.len());

        words.par_iter()
             .weight_max()
             .map(|word| TfIdfDefault::tfidf(word, &doc, all_docs.iter()))
             .collect_into(&mut term_frequencies);

        term_frequency_matrix.push(term_frequencies);
    }

    term_frequency_matrix
}

fn bool_to_f64(b: bool) -> f64 {
    if b {
        1f64
    } else {
        0f64
    }
}

pub fn check_for_code(self_texts: &[&str]) -> Vec<Vec<f64>> {
    lazy_static! {
        static ref FN_REGEX: Regex = Regex::new(r".*fn [:alpha:]{1}[:word:]*\(.*\)").expect("fn_regex");
        static ref LET_REGEX: Regex = Regex::new(r".*let( mut)? [:alpha:]{1}[:word:]*.* = .*;").expect("let_regex");
        static ref IF_LET_REGEX: Regex = Regex::new(r".*if let .* = match").expect("if_let_regex");
        static ref MACRO_REGEX: Regex = Regex::new(r".*[:alpha:]{1}[:word:]*! {0,1}[\{\(\[].*[\)\]\}]").expect("macro_regex");
    }

    self_texts.iter()
              .map(|text| {
                  vec![FN_REGEX.is_match(text),
                       LET_REGEX.is_match(text),
                       IF_LET_REGEX.is_match(text),
                       MACRO_REGEX.is_match(text)]
                      .into_iter()
                      .map(|b| bool_to_f64(b))
                      .collect()
              })
              .collect()
}

pub fn symbol_counts(self_texts: &[&str]) -> Vec<Vec<f64>> {
    let symbols = ['_', '-', ';', ':', '!', '?', '.', '(', ')', '[', ']', '{', '}', '*', '/',
                   '\\', '&', '%', '`', '+', '<', '=', '>', '|', '~', '$'];

    let mut freq_matrix = Vec::with_capacity(self_texts.len());

    let init_map: BTreeMap<&char, u64> = {
        let mut init_map = BTreeMap::new();
        for ch in symbols.iter() {
            init_map.insert(&*ch, 0);
        }
        init_map
    };

    for text in self_texts {
        let mut char_map = init_map.clone();

        for ch in text.chars() {
            if let Some(f) = char_map.get_mut(&ch) {
                *f += 1;
            }
        }

        let mut freq_vec = Vec::with_capacity(symbols.len());

        for symbol in symbols.iter() {
            let symbol_count = *char_map.get(symbol).unwrap_or(&0);
            freq_vec.push(symbol_count as f64);
        }

        freq_matrix.push(freq_vec);
    }
    freq_matrix
}

pub fn interesting_word_freq(self_texts: &[&str], spec_words: &[String]) -> Vec<Vec<f64>> {

    let mut freq_matrix = Vec::with_capacity(self_texts.len());
    let text_words: Vec<Vec<String>> = self_texts.iter()
                                                 .map(|t| tfidf_helper::get_words(*t))
                                                 .collect();

    let init_map: BTreeMap<String, u64> = {
        let mut init_map = BTreeMap::new();
        for word in spec_words {
            init_map.insert(word.to_owned(), 0);
        }
        init_map
    };

    for words in text_words.iter() {
        let mut freq_map: BTreeMap<String, u64> = init_map.clone();



        for word in words {
            if let Some(f) = freq_map.get_mut(&word.to_owned()) {
                *f += 1;
            }
        }

        let freq_vec: Vec<_> = freq_map.into_iter()
                                       .collect::<Vec<(_, u64)>>()
                                       .iter()
                                       .map(|t| t.1 as f64)
                                       .collect();

        freq_matrix.push(freq_vec);
    }

    freq_matrix
}

pub fn subs_to_float(subs: &[&str]) -> Vec<f64> {
    let mut sub_float_map = BTreeMap::new();
    let mut sub_floats = Vec::with_capacity(subs.len());
    let mut cur_sub = 0;

    for sub in subs {
        let f = *sub_float_map.entry(sub).or_insert_with(|| {
            let c = cur_sub;
            cur_sub = c + 1;
            c
        });
        sub_floats.push(f);
    }
    let sub_floats = sub_floats.into_iter()
                               .map(|f| f as f64)
                               .collect();
    // println!("{:?}", sub_floats);
    sub_floats
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_code_search() {
        let texts = vec!["Hey I need help with this function: pub fn get_stuff1(thing: &str) -> \
                          String {
                          let mut x: Option<_> = {Some(\"False\")};

                          if let Some(s) = match x {
                              println!(\"{:?}\", s);
                          }

                          return x;
                        }"];
        let r = check_for_code(&texts[..]);
        println!("{:?}", r);
    }

    #[test]
    fn test_word_freq() {
        let texts = vec!["the lazy brown fox jumped quickly = over the lazy fence"];
        let interesting_words = vec!["fence".to_owned(),
                                     "juniper".to_owned(),
                                     "lazy".to_owned(),
                                     "orange".to_owned(),
                                     "quickly".to_owned()];

        let expected = vec![1f64, 0f64, 2f64, 0f64, 1f64];

        let frequencies = interesting_word_freq(&texts[..], &interesting_words[..]);

        assert_eq!(expected, frequencies[0]);
    }
}

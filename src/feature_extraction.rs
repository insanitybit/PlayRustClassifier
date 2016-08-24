use rayon::prelude::*;
use regex::Regex;
// use rsml::tfidf_helper::*;
// use tfidf::{TfIdf, TfIdfDefault};
use std::collections::BTreeMap;
use std::ascii::AsciiExt;

pub fn convert_is_self(b: bool) -> f32 {
    if b {
        0f32
    } else {
        1f32
    }
}

pub fn convert_author_to_popularity<T: AsRef<str>>(authors: &[T],
                                                   rust_authors: &[&str])
                                                   -> Vec<f32> {
    let mut auth_count: BTreeMap<&str, _> = BTreeMap::new();

    for author in authors {
        if rust_authors.contains(&author.as_ref()) {
            *auth_count.entry(author.as_ref()).or_insert(0) += 1;
        }
    }
    let mut freqs = Vec::with_capacity(authors.len());

    for author in authors {
        let author_freq = *auth_count.get(author.as_ref()).unwrap_or(&0);
        freqs.push(author_freq as f32);
    }
    freqs
}

// pub fn text_to_docs<T: AsRef<str>>(texts: &[&str]) -> Vec<Vec<(String, usize)>> {
//     let mut docs = Vec::with_capacity(texts.len());
//     texts.par_iter()
//          .map(|s| str_to_doc(s))
//          .collect_into(&mut docs);
//     docs
// }

// // TODO: This should probably return an ndarray
// pub fn tfidf_reduce_selftext(self_texts: &[&str],
//                              words: &[&str],
//                              docs: &[Vec<(String, usize)>])
//                              -> Vec<Vec<f32>> {
//
//     let docs: Vec<Vec<_>> = docs.iter()
//                                 .map(|doc| doc.iter().map(|t| (t.0.as_str(), t.1)).collect())
//                                 .collect();
//     let all_docs = docs.clone();
//
//     let mut term_frequency_matrix = Vec::with_capacity(self_texts.len());
//     println!("TFIDF over {:?} words and {} docs", words.len(), docs.len());
//
//     for doc in docs {
//         let mut term_frequencies: Vec<f32> = Vec::with_capacity(words.len());
//
//         words.par_iter()
//              .weight_max()
//              .map(|word| TfIdfDefault::tfidf(word, &doc, all_docs.iter()))
//              .collect_into(&mut term_frequencies);
//
//         term_frequency_matrix.push(term_frequencies);
//     }
//
//     term_frequency_matrix
// }

fn bool_to_f32(b: bool) -> f32 {
    if b {
        1f32
    } else {
        0f32
    }
}

pub fn check_for_code(self_texts: &[&str]) -> Vec<Vec<f32>> {
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
                      .map(|b| bool_to_f32(b))
                      .collect()
              })
              .collect()
}

fn depluralize(s: &str) -> &str {
    if s.chars().last().unwrap() == 's' {
        &s[..s.len() - 1]
    } else {
        s
    }
}

fn is_letter(c: char) -> bool {
    match c as u8 {
        97...122 => true,
        _ => false,
    }
}

fn should_replace(c: u8) -> bool {
    match c {
        b'.' => true,
        b'?' => true,
        b'!' => true,
        b',' => true,
        b':' => true,
        b';' => true,
        b'(' => true,
        b')' => true,
        b'{' => true,
        b'}' => true,
        b']' => true,
        b'[' => true,
        b'/' => true,
        b'=' => true,
        b'|' => true,
        b'~' => true,
        _ => false,
    }
}

fn should_drop(c: u8) -> bool {
    match c {
        b'\'' => true,
        b'\"' => true,
        b'`' => true,
        b'-' => true,
        b'_' => true,
        b'*' => true,
        b'&' => true,
        _ => false,
    }
}

pub fn symbol_counts(self_texts: &[&str]) -> Vec<Vec<f32>> {
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
            freq_vec.push(symbol_count as f32);
        }

        freq_matrix.push(freq_vec);
    }
    freq_matrix
}


pub fn get_words(sentence: &str) -> Vec<String> {
    // This whole function could easily be optimized by turning the sentence into a Vec<u8>.
    // We can, fo rour purposes, simply strip out all non-ascii characters, and then do in-place
    // replacements. This would incur only a single copy for the function..
    let cleaned = sentence;

    let cleaned: String = cleaned.chars()
                                 .filter(|c| c.is_ascii())
                                 .collect();

    let mut cleaned: Vec<u8> = cleaned.bytes().collect();

    for c in cleaned.as_mut_slice() {
        if should_replace(*c) {
            *c = b' ';
        }
    }

    let cleaned: Vec<_> = cleaned.into_iter()
                                 .filter(|c| !should_drop(*c))
                                 .collect();

    // We take in a &str and filter all non-ascii out, this is safe
    let cleaned = unsafe { String::from_utf8_unchecked(cleaned) };
    let cleaned = cleaned.to_lowercase();


    cleaned.split_whitespace()
           .filter(|s| 2 < s.len() && s.len() < 10)
           .filter(|s| s.chars().all(is_letter))
           .map(depluralize)
           .filter(|s| 2 < s.len())
           .map(String::from)
           .collect()
}


pub fn interesting_word_freq(self_texts: &[&str], spec_words: &[String]) -> Vec<Vec<f32>> {

    let mut freq_matrix = Vec::with_capacity(self_texts.len());
    let text_words: Vec<Vec<String>> = self_texts.iter()
                                                 .map(|t| get_words(*t))
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
                                       .map(|t| t.1 as f32)
                                       .collect();

        freq_matrix.push(freq_vec);
    }

    freq_matrix
}

pub fn subs_to_float(subs: &[&str]) -> Vec<f32> {
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
                               .map(|f| f as f32)
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
        assert_eq!(r[0], vec![1f32, 1f32, 1f32, 1f32]);
    }

    #[test]
    fn test_symbol_freq() {
        let texts = vec!["Hey I need help with this function: pub fn get_stuff1(thing: &str) -> \
                          String {
                          let mut x: Option<_> = {Some(\"False\")};

                          if let Some(s) = match x {
                              println!(\"{:?}\", s);
                          }

                          return x;
                        }"];
        let s = symbol_counts(&texts[..]);
        assert_eq!(s[0],
                   vec![2f32, 1f32, 3f32, 4f32, 1f32, 1f32, 0f32, 4f32, 4f32, 0f32, 0f32, 4f32,
                        4f32, 0f32, 0f32, 0f32, 1f32, 0f32, 0f32, 0f32, 1f32, 2f32, 2f32, 0f32,
                        0f32, 0f32]);
    }

    #[test]
    fn test_author_popularity() {
        let authors = vec!["steveklabnik", "staticassert", "illogiq", "illogiq"];
        let popularity = vec![1f32, 1f32, 2f32, 2f32];

        assert_eq!(convert_author_to_popularity(&authors[..], &authors[..]),
                   popularity);
    }

    #[test]
    fn test_word_freq() {
        let texts = vec!["the lazy brown fox jumped quickly = over the lazy fence"];
        let interesting_words = vec!["fence".to_owned(),
                                     "juniper".to_owned(),
                                     "lazy".to_owned(),
                                     "orange".to_owned(),
                                     "quickly".to_owned()];

        let expected = vec![1f32, 0f32, 2f32, 0f32, 1f32];

        let frequencies = interesting_word_freq(&texts[..], &interesting_words[..]);

        assert_eq!(expected, frequencies[0]);
    }
}

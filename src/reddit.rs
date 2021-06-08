// use hyper::Client;
use rayon::prelude::*;
use serde_json;
use serde_json::Value;
// use std::io::prelude::*;
// use tiny_keccak::Keccak;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RawPostFeatures {
    pub is_self: bool,
    pub author: String,
    pub url: String,
    pub downs: u64,
    pub ups: u64,
    pub score: u64,
    pub selftext: String,
    pub subreddit: String,
    pub title: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProcessedPostFeatures {
    /// 0 if self, 1 if not self
    pub is_self: f32,
    /// The popularity of the author relative to the dataset
    pub author_popularity: f32,
    /// The number of downvotes
    pub downs: f32,
    /// The number of upvotes
    pub ups: f32,
    /// The overall score of the post
    pub score: f32,
    /// Length of the postcharacters
    pub post_len: f32,
    /// Word frequency vector
    pub word_freq: Vec<f32>,
    /// symbol frequency vector
    pub symbol_freq: Vec<f32>,
    /// Matches against regexes for rust code
    pub regex_matches: Vec<f32>,
}

pub struct RedditClient {
    client: reqwest::blocking::Client,
}

fn feature_from_value(value: &serde_json::Value) -> RawPostFeatures {
    let obj = value.as_object().unwrap();

    match obj.get("data") {
        Some(ref data) => feature_from_value(data),
        None => serde_json::from_value(value.clone()).unwrap(),
    }
}

pub fn get_posts(data: Vec<serde_json::Value>) -> Vec<RawPostFeatures> {
    let mut raw_features = Vec::with_capacity(data.len());
    data.par_iter()
        .map(|data| feature_from_value(data))
        .collect_into_vec(&mut raw_features);
    raw_features
}

impl RedditClient {
    pub fn new() -> RedditClient {
        RedditClient {
            client: reqwest::blocking::Client::new(),
        }
    }

    pub fn get_raw_features_from_url(&mut self, url: &str) -> Vec<serde_json::Value> {
        let query = format!("{}.json", url);

        let res = self.client.get(&query).send().unwrap();

        let body = res.text().expect("Couldn't parse body text");

        let data: Value = serde_json::from_str(&body).unwrap();
        let data = data.as_array().expect("Expected array");
        let data = data[0]
            .as_object()
            .expect("Data1 should have been an object");
        let data = data.get("data").expect("Expected key data");
        let data = data.as_object().expect("Data2 should have been an object");

        let data = data.get("children").expect("Expected children data");
        let data = data.as_array().unwrap();
        data.clone()
    }

    pub fn get_raw_features(
        &mut self,
        sub: &str,
        limit: u32,
        after: &Option<String>,
    ) -> (Vec<serde_json::Value>, Option<String>) {
        let query = match *after {
            Some(ref a) => {
                format!(
                    "https://www.reddit.com/r/{}/new.json?sort=new&limit={}&after={}",
                    sub, limit, a
                )
            }
            None => {
                format!(
                    "https://www.reddit.com/r/{}/new.json?sort=new&limit={}",
                    sub, limit
                )
            }
        };

        let res = self.client.get(&query).send().unwrap();

        let body = res.text().expect("Couldn't parse body text");

        let data: Value = serde_json::from_str(&body).unwrap();
        let data = data.as_object().unwrap();
        let data = data.get("data").unwrap();
        let data = data.as_object().unwrap();
        let after = data.get("after").unwrap();

        let data = data.get("children").unwrap();
        let data = data.as_array().unwrap();

        if after.is_string() {
            (data.clone(), after.as_str().map(|s| s.to_owned()))
        } else {
            (data.clone(), None)
        }
    }
}

pub fn anonymize_author(author: &str, iter: u64, key: &[u8]) -> String {
    use tiny_keccak::Hasher;
    let mut sha3 = tiny_keccak::Sha3::v512();

    let mut res: [u8; 512] = [0; 512];
    let authbytes: Vec<u8> = From::from(author);

    sha3.update(&authbytes);
    sha3.update(&key);
    sha3.finalize(&mut res);

    for _ in 0..iter {
        let mut sha3 = tiny_keccak::Sha3::v512();
        sha3.update(&res);
        sha3.update(&key);
        sha3.finalize(&mut res);
    }
    res.iter()
        .take(16)
        .map(|byte| format!("{:02x}", byte))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anon() {
        let anon = anonymize_author("name", 2, &b"key"[..]);
        println!("{:?}", anon);
    }
}

use hyper::Client;
use rayon::prelude::*;
use serde_json;
use serde_json::Value;
use std::io::prelude::*;
use stopwatch::Stopwatch;
use tiny_keccak::Keccak;

#[derive(Deserialize, Debug, Clone, RustcEncodable, RustcDecodable)]
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
    /// Word frequency vector
    pub word_freq: Vec<f64>,
}

pub struct RedditClient {
    client: Client,
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
        .collect_into(&mut raw_features);
    raw_features
}

impl RedditClient {
    pub fn new() -> RedditClient {
        RedditClient { client: Client::new() }
    }

    pub fn get_raw_features(&mut self,
                            sub: &str,
                            limit: u32,
                            after: &Option<String>)
                            -> (Vec<serde_json::Value>, Option<String>) {
        let query = match after {
            &Some(ref a) => {
                format!("https://www.reddit.com/r/{}/new.json?sort=new&limit={}&after={}",
                        sub,
                        limit,
                        a)
            }
            &None => {
                format!("https://www.reddit.com/r/{}/new.json?sort=new&limit={}",
                        sub,
                        limit)
            }
        };

        let mut res = self.client.get(&query).send().unwrap();

        let body = {
            let mut s = String::new();
            let _ = res.read_to_string(&mut s);
            s
        };

        let data: Value = serde_json::from_str(&body).unwrap();
        let data = data.as_object().unwrap();
        let data = data.get("data").unwrap();
        let data = data.as_object().unwrap();
        let after = data.get("after").unwrap();

        let data = data.get("children").unwrap();
        let data = data.as_array().unwrap();

        if after.is_string() {
            (data.clone(), after.as_string().map(|s| s.to_owned()))
        } else {
            (data.clone(), None)
        }


    }
}

pub fn anonymize_author(author: &str, iter: u64, key: &[u8]) -> String {
    let mut sha3 = Keccak::new_sha3_512();

    let mut res: [u8; 512] = [0; 512];
    let authbytes: Vec<u8> = From::from(author);

    sha3.update(&authbytes);
    sha3.update(&key);
    sha3.finalize(&mut res);

    for _ in 0..iter {
        let mut sha3 = Keccak::new_sha3_512();
        sha3.update(&res);
        sha3.update(&key);
        sha3.finalize(&mut res);
    }
    res.iter().take(16).map(|byte| format!("{:02x}", byte)).collect()
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

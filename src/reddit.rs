extern crate serde_json;

use hyper::Client;
use serde_json::Value;
use std::io::prelude::*;

#[derive(Debug)]
pub struct RawPostFeatures {
    pub is_self: bool,
    pub author_name: String,
    pub url: String,
    pub downvotes: u64,
    pub upvotes: u64,
    pub score: u64,
    pub edited: bool,
    pub selftext: String,
    pub subreddit: String,
    pub title: String,
}

pub struct RedditClient {
    client: Client,
}

impl RedditClient {
    pub fn new() -> RedditClient {
        RedditClient {
            client: Client::new(),
        }
    }

    pub fn get_raw_features(&mut self, sub: &str, limit: u32) -> Vec<RawPostFeatures> {
        let mut client = &mut self.client;
        let mut res = client.get(&format!("https://www.reddit.com/r/{}/new.json?sort=new&limit={}", sub, limit)).send().unwrap();

        let body = {
            let mut s = String::new();
            res.read_to_string(&mut s);
            s
        };

        let data: Value = serde_json::from_str(&body).unwrap();
        println!("{:?}", data);
        // let data = data.as_array().unwrap();
        let data = data.as_object().unwrap();
        let data = data.get("data").unwrap();
        let data = data.as_object().unwrap();
        let data = data.get("children").unwrap();
        let data = data.as_array().unwrap();

        let mut raw_features : Vec<RawPostFeatures> = Vec::with_capacity(data.len());

        let posts : Vec<_> = data.iter()
                                .map(|v| v.as_object().unwrap())
                                .map(|v| v.get("data").unwrap())
                                .map(|v| v.as_object().unwrap())
                                .collect();
        for post in posts {
            let feat =
            RawPostFeatures {
                is_self: post.get("is_self").unwrap().as_boolean().unwrap(),
                author_name: post.get("author").unwrap().as_string().unwrap().to_owned(),
                url: post.get("url").unwrap().as_string().unwrap().to_owned(),
                downvotes: post.get("downs").unwrap().as_u64().unwrap(),
                upvotes: post.get("ups").unwrap().as_u64().unwrap(),
                score: post.get("score").unwrap().as_u64().unwrap(),
                edited: post.get("edited").unwrap().as_boolean().unwrap().to_owned(),
                selftext: post.get("selftext").unwrap().as_string().unwrap().to_owned(),
                subreddit: post.get("subreddit").unwrap().as_string().unwrap().to_owned(),
                title: post.get("title").unwrap().as_string().unwrap().to_owned(),
            };
            raw_features.push(feat);
        }
        raw_features
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one() {
        let mut client = RedditClient::new();
        let features = client.get_raw_features("rust", 2);
    }
}

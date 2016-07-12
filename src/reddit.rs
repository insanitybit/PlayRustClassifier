use hyper::Client;
use rayon::prelude::*;
use serde_json;
use serde_json::Value;
use std::io::prelude::*;
use stopwatch::Stopwatch;

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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one() {
        let mut client = RedditClient::new();
        let mut posts = Vec::new();
        let (mut features, mut after) = client.get_raw_features("playrust", 100, &None);
        posts.extend_from_slice(&features[..]);
        loop {
            let res = client.get_raw_features("playrust", 100, &after);
            features = res.0;
            after = res.1;
            println!("{:?}", after);
            posts.extend_from_slice(&features[..]);
            if let None = after {
                break;
            }
        }
        println!("{:?}", posts.len());
        get_posts(posts);

    }
}

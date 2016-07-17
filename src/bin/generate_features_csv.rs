extern crate csv;
extern crate clap;
extern crate playrust_alert;
extern crate tiny_keccak;

use clap::{Arg, App};
use playrust_alert::reddit::RedditClient;
use playrust_alert::reddit::{get_posts, anonymize_author};
use tiny_keccak::Keccak;

fn get_args() -> String {
    let matches = App::new("Reddit Feature Generator")
                      .version("1.0")
                      .about("Collects posts from a subreddit")
                      .arg(Arg::with_name("subreddit")
                               .help("The subreddit to scrape")
                               .required(true)
                               .index(1))
                      .get_matches();

    matches.value_of("subreddit").unwrap().to_owned()
}



fn main() {
    let mut client = RedditClient::new();
    let sub = get_args();

    let mut raw_posts = Vec::with_capacity(1000);

    let after = None;
    loop {
        let (features, after) = client.get_raw_features(&sub, 100, &after);
        raw_posts.extend_from_slice(&features[..]);
        if after.is_none() {
            break;
        }
    }

    let mut wtr = csv::Writer::from_file(format!("./{}.csv", sub)).unwrap();

    let posts = get_posts(raw_posts);

    for record in posts.into_iter() {
        let _ = wtr.encode(record);
    }
}

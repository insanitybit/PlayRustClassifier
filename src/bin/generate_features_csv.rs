extern crate clap;
extern crate csv;
extern crate playrust_alert;
extern crate tiny_keccak;

use clap::{App, Arg};
use playrust_alert::reddit::get_posts;
use playrust_alert::reddit::RedditClient;

fn get_args() -> String {
    let matches = App::new("Reddit Feature Generator")
        .version("1.0")
        .about("Collects posts from a subreddit")
        .arg(
            Arg::with_name("subreddit")
                .help("The subreddit to scrape")
                .required(true)
                .index(1),
        )
        .get_matches();

    matches.value_of("subreddit").unwrap().to_owned()
}

fn main() {
    let mut client = RedditClient::new();
    let sub = get_args();

    // let mut raw_posts = Vec::with_capacity(1000);

    let mut wtr = csv::Writer::from_file(format!("./{}.csv", sub)).unwrap();
    let mut after = None;

    loop {
        println!("fetching");
        let (features, new_after) = client.get_raw_features(&sub, 100, &after);
        after = new_after;
        let posts = get_posts(features);
        for record in posts.into_iter() {
            let _ = wtr.encode(record);
        }

        if after.is_none() {
            break;
        }
    }
}

import time
current_milli_time = lambda: int(round(time.time() * 1000))

def author_popularity(authors, rust_authors):
    auth_count = {}
    for author in rust_authors:
        auth_count[author] = 0.0


    for author in authors:
        if auth_count.get(author):
            auth_count[author] += 1.0

    return [auth_count.get(author, 0.0) for author in authors]

def bench_author():
    with open('../data/rust_author_list') as f:
        rust_authors = f.read().split('\n')
        rust_authors.pop()

    with open('../data/all_authors') as f:
        authors = f.read().split('\n')
        authors.pop()


    start = current_milli_time()
    for _ in range(0, 1000):
        author_popularity(authors, rust_authors)

    end = current_milli_time()

    print("author:", end - start)


def bench_special_words():
    with open('../static_data/words_of_interest') as f:
        words_of_interest = f.read().split('\n')
        words_of_interest.pop()

    with open('../data/text_words') as f:
        text_words = f.read().split('\n')
        text_words.pop()

if __name__ == '__main__':
    bench_author()

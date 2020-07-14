import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


# given directory and return dictionary in the following form
# { "2.html": {"1.html", "3.html"}}
def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    out = dict()
    # set of pages the page link to
    links = corpus[page]

    if len(links) == 0:
        for k, v in corpus.items():
            out[k] = damping_factor / len(corpus)
    else:
        for k, v in corpus.items():
            if k in links:
                out[k] = damping_factor / len(links) + (1 - damping_factor) / len(corpus)
            else:
                out[k] = round(((1 - damping_factor) / len(corpus)), 4)
    return out


# return a dictionary where the keys are each page name and the values PageRank
# corpus mapping page name to a set of all pages linked to by that page
# (a number between 0 and 1).
def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    out = dict()

    results = dict()
    for k, v in corpus.items():
        results[k] = 0

    random_page = random.choice(list(corpus.keys()))

    for k, v in results.items():
        if k == random_page:
            results[k] += 1

    sequence = transition_model(corpus, random_page, damping_factor)

    for i in range(n-1):

        next_page = random.choices(list(sequence.keys()), list(sequence.values()))[0]

        for k, v in results.items():
            if k == next_page:
                results[k] += 1
        sequence = transition_model(corpus, next_page, damping_factor)

    for k, v in results.items():
        out[k] = v / n

    return out


# return a dictionary where the keys are each page name and the values PageRank
# (a number between 0 and 1).
def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    out = dict()

    for k, v in corpus.items():
        out[k] = 1 / len(corpus)

    return iterate(corpus, out)


def iterate(corpus, values):

    global temp
    temp = dict()

    for k, v in values.items():
        summation = 0
        for key, val in corpus.items():
            # for every page i links to p
            # A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself)
            if len(val) == 0:
                summation += values[key] / len(corpus)
            else:
                if k in val:
                    summation += values[key] / len(val)

        temp[k] = (1 - DAMPING) / len(corpus) + DAMPING * summation

    prev = list(values.values())
    current = list(temp.values())
    diff = []
    for i in range(len(prev)):
        diff.append(abs(prev[i] - current[i]))

    if max(diff) > 0.001:
        iterate(corpus, temp)

    return temp


if __name__ == "__main__":
    main()

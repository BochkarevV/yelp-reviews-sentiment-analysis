# %%
import os
import json
import zipfile
import pandas as pd


# %%
def fetch_per_category(n, path=os.path.join('data', 'yelp_academic_dataset_review.json.zip')):
    """
    Fetch the first N reviews per star rating from the YELP reviews dataset.
    The resulting set will consist of 5*N records.

    :param n: int
        Number per star rating of records to fetch.
    :param path: string, optional (default='data/yelp_academic_dataset_review.json.zip')
        Path to the source YELP reviews .json.zip archive.

    :return: list
        List of YELP review objects.
    """

    subsample = []
    counts = {}

    # Read zipped JSON
    with zipfile.ZipFile(path, 'r') as z:
        for filename in z.namelist():
            with z.open(filename) as f:

                # Iterate over the reviews
                for line in f:
                    review = json.loads(line.decode('utf-8'))

                    # Collect records and update the count
                    if review['stars'] not in counts:
                        subsample.append(review)
                        counts[review['stars']] = 1
                    elif counts[review['stars']] < n:
                        subsample.append(json.loads(line.decode('utf-8')))
                        counts[review['stars']] += 1

                    # Break when n records are gathered for all star ratings
                    if all(c == n for c in counts.values()) == n:
                        break

    return subsample


def save_subsample(dataset, path=os.path.join('data', 'yelp_reviews.json.gz')):
    """
    Save YELP data subsample into a compressed JSON.

    :param dataset: list
        List of YELP review objects.
    :param path: string, optional (default='data/yelp_reviews.json.gz')
        Path to save the subsample to.
    """

    df = pd.DataFrame(dataset)
    df.to_json(path, orient='records', compression='gzip', lines=True)


# %%
data = fetch_per_category(25_000)
save_subsample(data)

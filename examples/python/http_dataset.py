"""``HttpDataset``: one HTTP GET per item, made in Rust without holding the GIL. The URLs come
from a template and columns of values; every request carries a header; transient failures are
retried. The example starts a small local server, so it runs offline. Needs nothing but
dataloader-rs.

Run it with ``python examples/python/http_dataset.py``.
"""

from collections import Counter

from local_server import serve

from dataloader_rs import HttpDataset, PyDataloader

TOKEN = "Bearer example-token"
hits = Counter()


def answer(path: str, headers) -> tuple[int, bytes]:
    """``/shards/<shard>/items/<id>.txt`` returns a line of text. Requests without the token
    are refused, and item 5 fails twice with "503 Service Unavailable" before it succeeds."""
    if headers.get("Authorization") != TOKEN:
        return 401, b"missing token"
    parts = path.split("/")
    if len(parts) != 5 or parts[1] != "shards" or parts[3] != "items":
        return 404, b"not found"
    shard, item = parts[2], parts[4].removesuffix(".txt")
    hits[item] += 1
    if item == "5" and hits[item] <= 2:
        return 503, b"busy, try again"
    return 200, f"item {item} of shard {shard}".encode()


def main() -> None:
    with serve(answer) as base:
        # Item i is fetched from the URL made with the i-th value of each column. Strings are
        # percent-encoded ("shard a" becomes "shard%20a"); integers are written in decimal.
        # The template is parsed once, and each URL is made in Rust when its item is fetched.
        dataset = HttpDataset.from_template(
            base + "/shards/{shard}/items/{id}.txt",
            {"shard": ["shard a"] * 4 + ["shard b"] * 4, "id": list(range(8))},
            headers={"Authorization": TOKEN},
            timeout=5.0,  # seconds, for each attempt, the body included
            retries=3,  # up to 3 more attempts after a 503, a timeout or a lost connection
            backoff=0.01,  # waiting 10 ms, then 20 ms, then 40 ms (with some randomness)
        )
        loader = PyDataloader(
            dataset,
            batch_size=4,
            num_workers=2,
            item_concurrency=4,  # all 4 items of a batch in flight at once
            transform=lambda index, body: body.decode(),  # items arrive as bytes
        )
        for batch in loader:
            print(batch)
        print(f"item 5 took {hits['5']} requests: two 503 answers, then a success")

        # A failure that is not transient (here a 404) is not retried, and fails only its own
        # batch: the exception is raised for that batch, and the next one still comes.
        urls = [base + "/no/such/item", base + "/shards/c/items/0.txt"]
        dataset = HttpDataset(urls, headers={"Authorization": TOKEN})
        iterator = iter(PyDataloader(dataset, batch_size=1))
        for url in urls:
            try:
                print(f"{url}: {len(next(iterator)[0])} bytes")
            except OSError as error:
                print(f"{url}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()

"""``FileDataset`` and ``ObjectStoreDataset``: items read in Rust from local files, and from an
object store (S3, GCS, Azure or any HTTP server). Here the object store is a local directory
served over HTTP, so the example runs offline. Needs nothing but dataloader-rs.

Run it with ``python examples/python/files_and_objects.py``.
"""

import functools
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dataloader_rs import FileDataset, ObjectStoreDataset, PyDataloader

FOLDERS = ["cats", "cats", "cats", "dogs", "dogs", "dogs"]


def to_text(index: int, data: bytes) -> str:
    return data.decode()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass  # no line per request on the console


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index, folder in enumerate(FOLDERS):
            (root / folder).mkdir(exist_ok=True)
            (root / folder / f"{index}.txt").write_text(f"{folder} #{index}")

        # Files: a list of paths, or a template. "{folder}" takes the item's value in the
        # "folder" column and "{index}" is the item's index, so item 4 is <root>/dogs/4.txt.
        # Braces in the root itself are doubled so that they are not read as placeholders. A
        # value cannot contain "/" or be "..", so it can never point outside the directory.
        template = str(root).replace("{", "{{").replace("}", "}}") + "/{folder}/{index}.txt"
        files = FileDataset.from_template(
            template,
            {"folder": FOLDERS},
            timeout=10.0,  # on a network file system: a time limit for each read...
            retries=2,  # ...and up to 2 more attempts after a time-out
        )
        loader = PyDataloader(files, batch_size=3, item_concurrency=3, transform=to_text)
        for batch in loader:
            print("files:  ", batch)

        # Objects: the same directory, served over HTTP, stands in for a bucket. With S3 the
        # URL would be "s3://bucket/prefix", with credentials from the environment (AWS_*)
        # and settings in `options`, such as {"aws_region": "eu-west-3"}.
        handler = functools.partial(QuietHandler, directory=tmp)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            objects = ObjectStoreDataset(
                f"http://127.0.0.1:{server.server_port}/dogs",
                keys=["3.txt", "4.txt", "5.txt"],  # relative to the URL's prefix, "dogs"
                options={"allow_http": "true"},  # plain HTTP is refused unless allowed
                retries=3,  # the store retries 5xx answers, time-outs and lost connections
            )
            loader = PyDataloader(objects, batch_size=3, num_workers=1, transform=to_text)
            for batch in loader:
                print("objects:", batch)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()

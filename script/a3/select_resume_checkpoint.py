"""Select the latest trusted durable checkpoint, falling back after corruption."""

import argparse
import sys

from script.a3.checkpoint_store import select_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    path, payload, failures = select_checkpoint(args.directory)
    for failure in failures:
        print(f"Rejected checkpoint: {failure}", file=sys.stderr)
    print(f"Verified global_step={payload['state'].global_step}", file=sys.stderr)
    print(path.resolve())


if __name__ == "__main__":
    main()

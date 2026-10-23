"""Turn bench.py RESULT lines into the reStructuredText tables in README.rst.

    python bench.py ... | tee sweep.log
    python make_tables.py sweep.log
"""

import json
import sys

COLS = [
    ("n", "Volume"),
    ("ncz", "ncz"),
    ("t_class", "Setup (s)"),
    ("t_rec", "Recon (s)"),
    ("gvox", "Gvoxel/s"),
    ("peak_gpu_gb", "GPU (GB)"),
    ("peak_rss_gb", "Host (GB)"),
]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("RESULT "):
                r = json.loads(line[len("RESULT "):])
                r["gvox"] = round(r["n"] ** 3 / r["t_rec"] / 1e9, 2)
                r["n"] = f"{r['n']}^3"
                rows.append(r)
    return rows


def table(rows):
    """A simple RST grid table, columns padded to the widest cell."""
    head = [h for _, h in COLS]
    body = [[str(r.get(k, "")) for k, _ in COLS] for r in rows]
    w = [max(len(c) for c in col) for col in zip(head, *body)]
    rule = "  ".join("=" * x for x in w)
    fmt = lambda cells: "  ".join(c.ljust(x) for c, x in zip(cells, w)).rstrip()
    return "\n".join([rule, fmt(head), rule, *map(fmt, body), rule])


def main():
    rows = load(sys.argv[1])
    for alg in ("fourierrec", "lprec"):
        for dt in ("float32", "float16"):
            sel = [r for r in rows if r["algorithm"] == alg and r["dtype"] == dt]
            if not sel:
                continue
            title = f"{alg}, {dt}"
            print(f"\n{title}\n{'~' * len(title)}\n")
            print(table(sel))


if __name__ == "__main__":
    main()

"""Extract a library paper to text (or page images) for source checking.

The publication-fidelity workflow (dopli LLM_PRACTICE Rule 1, adopted here)
re-reads a source before any methods text is written or revised, which means
turning a PDF in [library/](../../library/) into something greppable. The Read
tool cannot open PDFs in this environment, so extraction goes through pymupdf;
this script is the ported dopli helper (`dopli/testbed/scripts/
extract_paper.py`) with this repo's layout: PDFs sit in ``library/`` itself
and ``library/index.md`` abbreviates filenames and may list several bibkeys
per row.

Usage (blessed conda invocation, Bash tool)::

    conda run -n UTESpac_Plus --no-capture-output python testbed/scripts/extract_paper.py Thomas2007

    python extract_paper.py Thomas2007              # by bibkey
    python extract_paper.py "Turner and Leclerc"    # by filename fragment
    python extract_paper.py Kaimal1972 --pages 1-3  # limit the range
    python extract_paper.py Turner1994 --render     # scanned: PNGs to read
    python extract_paper.py --check-index           # directory vs index.md

Text lands in ``library/extracted/<bibkey>.txt`` with ``===== PAGE n =====``
markers, so a grep hit can be mapped back to a page. Page images land beside
it as ``<bibkey>_p<n>.png``, and a ``--pages`` run writes
``<bibkey>_p<spec>.txt`` so it cannot stand in for the whole paper. The
extractions are gitignored with the PDFs and accumulate as a local, greppable
cache across sessions -- check there before re-extracting. Grep with
``grep -a`` (extractions can trip ripgrep's binary detection).

Three things this handles that a bare snippet does not:

* **Scanned PDFs.** Antonia 1979, Nieuwstadt 1984, and Turner & Leclerc 1994
  carry no usable text layer. Extraction reports characters-per-page and says
  so rather than silently writing a file of page headers; ``--render`` then
  produces images to read visually.
* **Filename resolution.** Paths in ``library/`` contain spaces and non-ASCII
  punctuation (U+2019 in Desjardins', U+2010 in Kaimal 1972), which is why
  passing them through a shell has to be avoided. Resolution goes through
  ``index.md``'s bibkey column (rows may hold several comma-separated keys),
  falling back to a case-insensitive fragment match on the filename.
* **Index drift.** ``--check-index`` diffs the directory against ``index.md``
  in both directions, which is how a user-supplied paper is noticed (per the
  standing rule, never by mtime and never by searching outside the repo).
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "library"
INDEX = PAPERS / "index.md"
DEFAULT_OUT = PAPERS / "extracted"

# below this, a page's text layer is headers/watermark only -- treat as scanned
THIN_PAGE_CHARS = 200


def _norm(text):
    """Fold the punctuation variants that differ between index.md and disk."""
    text = unicodedata.normalize("NFKC", text)
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u2010", "-"),
                 ("\u2013", "-"), ("\u2014", "-")):
        text = text.replace(a, b)
    return text.lower()


def read_index():
    """Parse index.md into a list of (name_stub, [bibkeys], note) rows.

    The filename column abbreviates: trailing ``...`` truncation and a
    parenthetical note may follow the name, and the bibkey column may hold
    several comma-separated keys (book chapters). The stub returned is the
    literal leading fragment usable for prefix matching against disk names.
    """
    rows = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= {"-", " "} or cells[0] == "File (abbreviated)":
            continue
        stub = re.sub(r"\s*\(.*", "", cells[0]).rstrip(". ").strip()
        keys = [k.strip() for k in re.split(r"[,;]", cells[1])
                if re.fullmatch(r"\w+", k.strip())]
        rows.append((stub, keys, cells[2] if len(cells) > 2 else ""))
    return rows


def _stub_matches(stub, name):
    """Abbreviated index stub vs normalized disk name.

    ``...`` may truncate the stub anywhere (leading, mid, trailing); the
    literal fragments must appear in the name in order, the first one at the
    start unless the stub itself starts with ``...``.
    """
    frags = [_norm(f) for f in stub.split("...") if f.strip()]
    if not frags:
        return False
    pos = 0
    for i, frag in enumerate(frags):
        found = name.find(frag, pos)
        if found < 0 or (i == 0 and not stub.startswith("...") and found != 0):
            return False
        pos = found + len(frag)
    return True


def _disk_match(stub, on_disk):
    """Abbreviated index stub -> disk path, unique match or None."""
    hits = [p for n, p in on_disk.items() if _stub_matches(stub, n)]
    return hits[0] if len(hits) == 1 else None


def resolve(query):
    """Query (bibkey or filename fragment) -> (path, bibkey)."""
    rows = read_index()
    on_disk = {_norm(p.name): p for p in PAPERS.glob("*.pdf")}

    for stub, keys, _ in rows:
        for bibkey in keys:
            if bibkey.lower() == query.lower():
                path = _disk_match(stub, on_disk)
                if path is None:
                    raise SystemExit(
                        f"index.md lists {bibkey} as {stub!r} but no unique "
                        f"file in {PAPERS} matches -- the PDFs are gitignored, "
                        f"so this is a local-copy problem, not an index error")
                return path, bibkey

    hits = [p for key, p in on_disk.items() if _norm(query) in key]
    if len(hits) == 1:
        path = hits[0]
        bibkey = next((k[0] for s, k, _ in rows
                       if k and _stub_matches(s, _norm(path.name))),
                      path.stem[:40])
        return path, bibkey
    if not hits:
        raise SystemExit(f"no paper matches {query!r}; try --check-index")
    raise SystemExit(
        f"{query!r} matches {len(hits)} papers:\n  "
        + "\n  ".join(sorted(p.name for p in hits)))


def parse_pages(spec, n_pages):
    if not spec:
        return range(n_pages)
    lo, _, hi = spec.partition("-")
    lo = int(lo) - 1
    hi = int(hi) if hi else lo + 1
    return range(max(lo, 0), min(hi, n_pages))


def extract(path, bibkey, out_dir, pages=None, render=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(path)
    idx = parse_pages(pages, len(doc))

    chunks, thin = [], []
    for i in idx:
        text = doc[i].get_text()
        if len(text.strip()) < THIN_PAGE_CHARS:
            thin.append(i + 1)
        chunks.append(f"\n===== PAGE {i + 1} =====\n{text}")

    # a page-limited run gets its own filename: the cache is consulted before
    # re-extracting, so a partial run must not stand in for the whole paper
    stem = bibkey if not pages else f"{bibkey}_p{pages}"
    txt_path = out_dir / f"{stem}.txt"
    txt_path.write_text("".join(chunks), encoding="utf-8")
    body = sum(len(c) for c in chunks)
    print(f"{bibkey}: {len(doc)} pages ({len(idx)} extracted), {body} chars")
    print(f"  text  -> {txt_path}")

    if thin:
        print(f"  WARNING: {len(thin)}/{len(idx)} pages have almost no text "
              f"layer (pages {thin[:8]}{'...' if len(thin) > 8 else ''}).")
        if not render:
            print("  Quotes cannot be taken from those pages. Re-run with "
                  "--render and read the images instead.")

    if render:
        mat = fitz.Matrix(2.0, 2.0)
        for i in idx:
            img = out_dir / f"{bibkey}_p{i + 1}.png"
            doc[i].get_pixmap(matrix=mat).save(img)
        print(f"  images -> {out_dir / (bibkey + '_p<n>.png')} "
              f"({len(idx)} pages)")
    doc.close()
    return txt_path


def check_index():
    """Diff the library PDFs against index.md, both directions."""
    rows = read_index()
    on_disk = {_norm(p.name): p.name for p in PAPERS.glob("*.pdf")}

    matched = set()
    missing_file = []
    for stub, keys, _ in rows:
        path = _disk_match(stub, on_disk={k: Path(v) for k, v in on_disk.items()})
        if path is None:
            missing_file.append(stub)
        else:
            matched.add(_norm(path.name))
    missing_row = sorted(on_disk[k] for k in on_disk.keys() - matched)

    print(f"index.md rows: {len(rows)};  PDFs on disk: {len(on_disk)}")
    if missing_row:
        print("\nOn disk, absent from index.md (a user-supplied paper needs a "
              "row, a references.bib entry, and a page-1 identity check):")
        for name in missing_row:
            print(f"  + {name}")
    if missing_file:
        print("\nIn index.md, absent from disk (PDFs are gitignored -- expected "
              "on a fresh clone):")
        for name in missing_file:
            print(f"  - {name}")
    if not missing_row and not missing_file:
        print("\nin sync")
    return 1 if missing_row else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("query", nargs="?",
                    help="bibkey (index.md column 2) or filename fragment")
    ap.add_argument("--pages", help="1-based page range, e.g. 3 or 4-9")
    ap.add_argument("--render", action="store_true",
                    help="also write one PNG per page (scanned PDFs, figures)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--check-index", action="store_true",
                    help="diff library/ PDFs against index.md and exit")
    args = ap.parse_args(argv)

    if args.check_index:
        return check_index()
    if not args.query:
        ap.error("give a bibkey/fragment, or --check-index")

    path, bibkey = resolve(args.query)
    extract(path, bibkey, args.out, pages=args.pages, render=args.render)
    return 0


if __name__ == "__main__":
    sys.exit(main())

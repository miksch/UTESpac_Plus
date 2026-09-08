"""card_convert.py
Convert Campbell Scientific binary card files (TOB3) to TOA5 with
csidft_convert.exe, so every table carries its full unit/aggregation
header rows.

Ported from an in-house card-conversion notebook (Card
Convert section). The converter executable and its DLL live in
raw_processing/utils/ (copied from the same in-house tree,
gitignored). Output files are named <table>_<yyyy_mm_dd_HHMMSS>.dat
using the file-creation timestamp decoded from the TOB3 header line.

Pull folders are named <PREFIX>_<yyyymmdd>_FLX by card-pull date; each
holds the data since the previous pull, so ``select_pull_dirs`` picks
folders whose covered interval overlaps a date window (a multi-week
flux file created before the window can still contain data inside it).
"""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

DEFAULT_EXE = Path(__file__).resolve().parent / "utils" / "csidft_convert.exe"

_PULL_DATE_RE = re.compile(r"_(\d{8})_")


def split_last_first(s):
    """Split *s* between its last letter and first digit.

    Returns (letters_part, digits_part); (*s*, '') when no boundary.
    """
    match = re.search(r"([a-zA-Z])(\d)", s)
    if match:
        return s[:match.start(1) + 1], s[match.start(2):]
    return s, ""


def header_timestamp(f):
    """File-creation timestamp from the first TOB3/TOA1 header line."""
    with open(f, "rb") as file:
        first_line = file.readline()
    return datetime.strptime(first_line[-22:-3].decode("utf-8"),
                             "%Y-%m-%d %H:%M:%S")


def process_file(f, out_dir, cmd_exe=DEFAULT_EXE):
    """Convert one binary file to TOA5 in *out_dir*.

    Output name: table stem (numeric suffix dropped) + header timestamp,
    so files sort chronologically.
    """
    stamp = header_timestamp(f).strftime("%Y_%m_%d_%H%M%S")
    stem_pt1, _ = split_last_first(f.stem)
    output_path = Path(out_dir, f"{stem_pt1}_{stamp}.dat")

    command = [str(cmd_exe), str(f), str(output_path), "toa5"]
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully for", f)
    except subprocess.CalledProcessError as e:
        print(f"Command failed for {f} with return code {e.returncode}")


def csi_card_convert(file_list, out_dir, cmd_exe=DEFAULT_EXE):
    """Convert *file_list* to TOA5 in *out_dir* on a thread pool."""
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda f: process_file(f, out_dir, cmd_exe),
                          file_list))


def select_pull_dirs(bin_dir, start_date, end_date):
    """Pull folders under *bin_dir* whose data overlap [start, end).

    Folder <PREFIX>_<yyyymmdd>_FLX covers (previous pull date, pull
    date]; the first folder covers everything up to its date. Folders
    without a yyyymmdd token are skipped.
    """
    dated = []
    for d in sorted(Path(bin_dir).iterdir()):
        if not d.is_dir():
            continue
        m = _PULL_DATE_RE.search(d.name)
        if m:
            dated.append((datetime.strptime(m.group(1), "%Y%m%d"), d))
    dated.sort()

    selected = []
    prev = datetime.min
    for pull_date, d in dated:
        if pull_date >= start_date and prev < end_date:
            selected.append(d)
        prev = pull_date
    return selected

import os
import gzip
import hashlib
from datetime import datetime
from typing import Iterable, List, Tuple, Optional


def _open_text_file(path: str, encoding: str = "utf-8", errors: str = "strict"):
    """Open a text file, supporting plain and gzip-compressed files.

    Returns a file-like object opened in text mode (iterator over lines).
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding=encoding, errors=errors)
    return open(path, "r", encoding=encoding, errors=errors)


def _read_lines(
    path: str, encoding: str = "utf-8", errors: str = "strict"
) -> Iterable[str]:
    with _open_text_file(path, encoding=encoding, errors=errors) as fh:
        for line in fh:
            yield line


def _stable_id(s: str) -> int:
    """Deterministic int id for a string using md5 (stable across runs)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (10**9)


def _assign_node_id(raw: str, registry: dict) -> int:
    """Map a raw identifier string to an int id, guarding against collisions.

    Numeric-looking identifiers are used as-is; everything else falls back to
    `_stable_id`'s md5-based hash. Both paths share one `registry` (raw string
    -> assigned id) so a collision is caught regardless of which path produced
    it -- including a hashed username landing on a numeric id used elsewhere.
    Does not change the id assigned to any non-colliding string (so a
    collision-free file, like the current wiki-RfA dump, encodes identically
    to before this guard existed); it only turns a *future* collision into a
    loud failure instead of a silent node merge.
    """
    try:
        node_id = int(raw)
    except (TypeError, ValueError):
        node_id = _stable_id(raw)

    existing_raw = registry.get(node_id)
    if existing_raw is not None and existing_raw != raw:
        raise RuntimeError(
            f"Node ID collision detected: distinct identifiers {existing_raw!r} "
            f"and {raw!r} both map to id={node_id}. Silently proceeding would "
            "merge two distinct users into one graph node. Refusing to build "
            "the dataset -- widen/replace the id scheme (e.g. a bijective "
            "interning table) before retrying."
        )
    registry[node_id] = raw
    return node_id


def load_wiki_rfa(cfg):
    """Load the Wiki-RfA dataset.

    Parses blocks separated by blank lines. Fields include `SRC:`, `TGT:`, `VOT:`,
    `YEA:` and `TXT:`. TXT may be multi-line. The loader returns a list of
    `(u, v, label)` tuples where `label` is the raw signed vote (e.g. -1, 0, 1)
    or mapped classes when not in binary mode.

    Binary mode (`cfg.dataset.binary=True`) will skip entries with VOT==0 and
    preserve the signed labels (-1 or 1).
    """

    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)

    # Read blocks
    blocks = []
    cur = []
    for line in _read_lines(path):
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line.rstrip("\n"))
    if cur:
        blocks.append(cur)

    parsed = []
    node_id_registry = {}
    for block in blocks:
        data = {}
        last_key = None
        for ln in block:
            if ":" in ln and ln.split(":", 1)[0].isupper():
                key, val = ln.split(":", 1)
                key = key.strip()
                val = val.lstrip()
                data[key] = val
                last_key = key
            else:
                if last_key is not None:
                    data[last_key] = data.get(last_key, "") + "\n" + ln

        src_raw = data.get("SRC")
        tgt_raw = data.get("TGT")
        vot_raw = data.get("VOT")
        # Prefer `DAT` field and `YEA` (year) field present in this dataset
        dat_raw = data.get("DAT")
        yea_raw = data.get("YEA")

        if not src_raw or not tgt_raw or vot_raw is None:
            continue

        # convert to numeric ids (collision-guarded -- see _assign_node_id)
        u = _assign_node_id(src_raw, node_id_registry)
        v = _assign_node_id(tgt_raw, node_id_registry)

        # normalize vote
        try:
            vot_val = int(float(vot_raw))
        except Exception:
            vr = vot_raw.strip().lower()
            if vr in ("support", "for", "yes", "+", "+1"):
                vot_val = 1
            elif vr in ("oppose", "against", "no", "-", "-1"):
                vot_val = -1
            else:
                vot_val = 0

        # parse DAT/YEA into deterministic ISO timestamp string when possible
        ts_val = None
        if dat_raw:
            # Examples observed: "23:13, 19 April 2013"
            # Try several common formats
            parsed_dt = None
            fmts = [
                "%H:%M, %d %B %Y",
                "%H:%M, %d %b %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y-%m-%d",
            ]
            for f in fmts:
                try:
                    parsed_dt = datetime.strptime(dat_raw.strip(), f)
                    break
                except Exception:
                    parsed_dt = None
            if parsed_dt is None and yea_raw:
                # try appending year from YEA if DAT lacks it
                try:
                    dat_with_year = f"{dat_raw.strip()} {yea_raw}"
                    for f in [
                        "%H:%M, %d %B %Y",
                        "%H:%M, %d %b %Y",
                        "%d %B %Y",
                        "%d %b %Y",
                    ]:
                        try:
                            parsed_dt = datetime.strptime(dat_with_year, f)
                            break
                        except Exception:
                            parsed_dt = None
                except Exception:
                    parsed_dt = None

            if parsed_dt is not None:
                # normalize to ISO (date only is sufficient for ordering)
                ts_val = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
        elif yea_raw:
            # fallback to year-only timestamp
            try:
                y = int(str(yea_raw).strip())
                ts_val = f"{y:04d}-01-01 00:00:00"
            except Exception:
                ts_val = None

        parsed.append((u, v, vot_val, ts_val))

    # Build final edges list
    binary = bool(getattr(cfg.dataset, "binary", False))
    edges_out = []
    if binary:
        for u, v, vot, ts in parsed:
            if vot == 0:
                continue
            edges_out.append((u, v, vot, ts))
    else:
        edges_out = [(u, v, vot, ts) for (u, v, vot, ts) in parsed]

    # Apply configurable post-processing (self-loops, multiedges)
    return postprocess_edges(cfg, edges_out)


def load_chess(cfg):
    """Load the Chess dataset from an edge list file (supports .gz)."""
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    edges = []
    for line in _read_lines(path):
        if line.startswith("%"):
            continue  # Skip comments
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # Invalid line
        try:
            u, v, label = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        edges.append((u, v, label))
    return postprocess_edges(cfg, edges)


def load_bitcoin(cfg):
    """Load the Bitcoin Alpha or OTC dataset from a CSV file (supports .gz).

    CSV columns: source, target, rating, timestamp
    Ratings are on a scale of -10 to +10.

    Binary mode (`cfg.dataset.binary=True`) maps positive ratings to +1,
    negative ratings to -1, and skips zero-rated edges.
    Non-binary mode preserves the raw integer rating.
    """
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    binary = bool(getattr(cfg.dataset, "binary", False))
    edges = []
    for line in _read_lines(path):
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue  # Invalid line
        try:
            u, v, rating = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        # Parse optional timestamp (4th column)
        ts = None
        if len(parts) >= 4:
            ts = parts[3].strip() or None
        if binary:
            if rating > 0:
                label = 1
            elif rating < 0:
                label = -1
            else:
                continue  # skip neutral
        else:
            label = rating
        edges.append((u, v, label, ts))
    return postprocess_edges(cfg, edges)


def load_toy(cfg):
    """Load the toy dataset from an edge list file (supports .gz)."""
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    edges = []
    for line in _read_lines(path):
        if line.startswith("%"):
            continue  # Skip comments
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # Invalid line
        try:
            u, v, label = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        edges.append((u, v, label))
    return postprocess_edges(cfg, edges)


def _load_signed_edge_list_simple(path: str):
    """Helper for simple signed edge-list files with optional header lines.

    Lines starting with `#` are ignored. Each data line is expected to have
    three columns: from, to, sign (e.g. -1 or 1). Columns can be tab- or
    whitespace-separated.
    """
    edges = []
    for line in _read_lines(path):
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            # try comma-separated fallback
            parts = ln.split(",")
            if len(parts) < 3:
                continue
        try:
            u = int(parts[0])
            v = int(parts[1])
            s = int(parts[2])
        except Exception:
            continue
        edges.append((u, v, s))
    return edges


def load_epinions(cfg):
    """Load Epinions signed social network (soc-sign-epinions).

    The file in `data/epinions` follows the standard `FromNodeId\tToNodeId\tSign`
    format with `#` headers. This loader returns `(u,v,sign)` tuples. Datasets
    in this collection are already binary/signed; if `cfg.dataset.binary` is
    True we still simply return signed edges (no remapping).
    """
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    parsed = _load_signed_edge_list_simple(path)
    # keep signed labels as-is (dataset is already signed)
    return postprocess_edges(cfg, parsed)


def load_slashdot(cfg):
    """Load Slashdot signed social network (soc-sign-Slashdot*).

    Same format as Epinions. Returns `(u,v,sign)` tuples with original signs.
    """
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    parsed = _load_signed_edge_list_simple(path)
    return postprocess_edges(cfg, parsed)


def load_wiki_elec(cfg):
    """Load the Wikipedia Elections (wiki-Elec) dataset.

    The file is a block-structured text file where each block describes one
    election:
      E  <1/0>          - election result (1=promoted, 0=not)
      T  <datetime>     - time election was closed
      U  <id> <name>    - candidate user id and username
      N  <id> <name>    - nominator user id and username
      V  <vote> <voter_id> <date> <time> <username>  - individual votes

    Each vote V creates a directed edge: voter_id → candidate_id (U), with the
    vote value as the label (1=support, 0=neutral, -1=oppose) and the V-line
    datetime as the timestamp.

    Binary mode (`cfg.dataset.binary=True`) skips neutral votes (vote == 0).
    """
    path = os.path.join(cfg.dataset.data_dir, cfg.dataset.edge_list_file)
    binary = bool(getattr(cfg.dataset, "binary", False))

    edges = []
    candidate_id = None

    for line in _read_lines(path, encoding="latin-1"):
        ln = line.rstrip("\n")
        # Skip blank lines and comment header lines that aren't data blocks
        if not ln.strip() or (ln.startswith("#") and not ln.startswith("#\t")):
            if ln.strip() == "":
                # blank line resets candidate context
                candidate_id = None
            continue

        parts = ln.split()
        if not parts:
            continue

        tag = parts[0]

        if tag == "U":
            # U <id> <username>
            try:
                candidate_id = int(parts[1])
            except (IndexError, ValueError):
                candidate_id = None

        elif tag == "V" and candidate_id is not None:
            # V <vote> <voter_id> <date> <time> <username>
            if len(parts) < 3:
                continue
            try:
                vote = int(parts[1])
                voter_id = int(parts[2])
            except ValueError:
                continue
            # Parse timestamp: date and time are in parts[3] and parts[4]
            ts = None
            if len(parts) >= 5:
                ts = f"{parts[3]} {parts[4]}"
            elif len(parts) >= 4:
                ts = parts[3]

            if binary and vote == 0:
                continue

            edges.append((voter_id, candidate_id, vote, ts))

        # E, T, N lines are ignored for edge construction

    return postprocess_edges(cfg, edges)


# 🔁 Registry of dataset loaders
DATASET_LOADERS = {
    "chess": load_chess,
    "bitcoin-alpha": load_bitcoin,
    "bitcoin-alpha-binary": load_bitcoin,
    "bitcoin-otc": load_bitcoin,
    "bitcoin-otc-binary": load_bitcoin,
    "wiki-rfa": load_wiki_rfa,
    "wiki-elec": load_wiki_elec,
    "toy": load_toy,
    "epinions": load_epinions,
    "slashdot090221": load_slashdot,
    "synthetic-fog": load_bitcoin,
}


def get_loader(name):
    """Return the loader function for a given dataset name."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Dataset '{name}' is not supported.")
    return DATASET_LOADERS[name]


def postprocess_edges(cfg, edges: List[Tuple]) -> List[Tuple[int, int, int]]:
    """Post-process edges according to `cfg.dataset` options.

    Inputs may be 3-tuples `(u,v,label)` or 4-tuples `(u,v,label,timestamp)`.
    Supported `cfg.dataset` options:
      - `remove_self_loops` (bool, default True)
      - `multiedge_handling` (str, one of 'keep','aggregate_majority','aggregate_sum','count','most_recent')

    Returns a list of 3-tuples `(u,v,label)`.
    """
    remove_self = bool(getattr(cfg.dataset, "remove_self_loops", True))
    multiedge_handling = getattr(cfg.dataset, "multiedge_handling", "keep")

    # Normalize entries to (u,v,label,ts)
    norm = []
    for e in edges:
        if len(e) == 3:
            u, v, lab = e
            ts = None
        elif len(e) >= 4:
            u, v, lab, ts = e[0], e[1], e[2], e[3]
        else:
            continue
        norm.append((u, v, lab, ts))

    if remove_self:
        norm = [(u, v, lab, ts) for (u, v, lab, ts) in norm if u != v]

    if multiedge_handling == "keep":
        return [(u, v, lab) for (u, v, lab, ts) in norm]

    # Build grouping
    agg = {}
    for u, v, lab, ts in norm:
        agg.setdefault((u, v), []).append((lab, ts))

    out = []
    for (u, v), lab_ts in agg.items():
        if multiedge_handling == "count":
            out.append((u, v, len(lab_ts)))
            continue

        if multiedge_handling == "aggregate_sum":
            try:
                s = sum(int(x) for x, _ in lab_ts)
                out.append((u, v, s))
            except Exception:
                out.append((u, v, lab_ts[-1][0]))
            continue

        if multiedge_handling == "aggregate_majority":
            try:
                s = sum(int(x) for x, _ in lab_ts)
                if s > 0:
                    out.append((u, v, 1))
                elif s < 0:
                    out.append((u, v, -1))
                else:
                    out.append((u, v, 0))
            except Exception:
                out.append((u, v, lab_ts[-1][0]))
            continue

        if multiedge_handling == "most_recent":
            # pick label of entry with most recent timestamp (lexicographic fallback)
            # entries without timestamps are considered older
            def _key(item):
                lab, ts = item
                if ts is None:
                    return ""
                return str(ts)

            # get max by _key; if all keys empty, fallback to last occurrence
            best = (
                max(lab_ts, key=_key)
                if any(ts is not None for _, ts in lab_ts)
                else lab_ts[-1]
            )
            out.append((u, v, best[0]))
            continue

        # Unknown handling -> pick last
        out.append((u, v, lab_ts[-1][0]))

    return out

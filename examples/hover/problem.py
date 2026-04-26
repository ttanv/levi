"""HoVer — faithful replication of gepa-artifact's hover benchmark.

Mirrors `gepa_artifact/benchmarks/hover/` line-for-line:
  - Dataset: HoVer train split, filtered to 3-hop examples (count_unique_docs == 3),
    shuffled with random.Random(0), then sliced 40/40/20 test/val/train and
    trimmed via random.Random(1).sample to (150 train, 300 val, 300 test).
  - Retriever: bm25s over `wiki.abstracts.2017.jsonl` (downloaded on first run),
    BM25(k1=0.9, b=0.4), tokenizer stopwords="en" + english Stemmer. Hop-1/2 use
    k=7, hop-3 uses k=10.
  - Program: 4 dspy.ChainOfThought modules with the same signature strings as
    `HoverMultiHop` (claim,passages->summary ; claim,summary_1->query ;
    claim,context,passages->summary ; claim,summary_1,summary_2->query).
  - Scoring: `discrete_retrieval_eval` — gold supporting-fact titles must be a
    subset of the union of titles retrieved across all 3 hops.

Levi optimizes the 4 instruction strings plugged into each predictor's signature
at eval time via `pred.signature.with_instructions(...)`. Seed strings are
empty => DSPy uses its default auto-generated instruction, matching GEPA's
starting point.
"""

from __future__ import annotations

import json
import os
import random
import tarfile
import threading
import urllib.request
from pathlib import Path

import bm25s
import dspy
import Stemmer

# Qwen3 official thinking-mode sampling params (HuggingFace model card):
# temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768.
TASK_MODEL = os.getenv("HOVER_TASK_MODEL", "openrouter/qwen/qwen3-8b")
TASK_PROVIDER_ONLY = os.getenv("HOVER_PROVIDER_ONLY", "alibaba")
TASK_TEMPERATURE = float(os.getenv("HOVER_TEMPERATURE", "0.6"))
TASK_TOP_P = float(os.getenv("HOVER_TOP_P", "0.95"))
TASK_TOP_K = int(os.getenv("HOVER_TOP_K", "20"))
TASK_MIN_P = float(os.getenv("HOVER_MIN_P", "0"))
TASK_MAX_TOKENS = int(os.getenv("HOVER_MAX_TOKENS", "32768"))
TASK_TIMEOUT = float(os.getenv("HOVER_TIMEOUT", "360"))
TASK_MAX_WORKERS = int(os.getenv("HOVER_MAX_WORKERS", "8"))

CACHE_DIR = Path(
    os.getenv("HOVER_CACHE_DIR", str(Path.home() / ".cache" / "levi" / "hover"))
)
CORPUS_URL = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
CORPUS_PATH = CACHE_DIR / "wiki.abstracts.2017.jsonl"
BM25_PATH = CACHE_DIR / "bm25s_retriever"

PROBLEM_DESCRIPTION = (
    "Optimize a 4-component prompt bundle for HoVer 3-hop fact retrieval. "
    "The DSPy program: summarize1 reads hop-1 BM25 passages, create_query_hop2 "
    "writes a follow-up query, summarize2 integrates hop-2 passages, "
    "create_query_hop3 writes the final query whose hop-3 retrieval (k=10) is "
    "added to the union of retrieved docs. Scoring: gold supporting-fact titles "
    "must be a subset of the union of retrieved doc titles across all 3 hops."
)

# Seed = empty -> DSPy uses the default auto-generated instruction for each
# signature (exactly what HoverMultiHop starts from in the artifact).
SEED_BUNDLE: dict[str, str] = {
    "summarize1": "",
    "create_query_hop2": "",
    "summarize2": "",
    "create_query_hop3": "",
}


# ---------- BM25 retriever (port of gepa_artifact/benchmarks/hover/hover_program.py) ----------

_retriever: bm25s.BM25 | None = None
_stemmer = None
_corpus: list[str] | None = None
_init_lock = threading.Lock()


def _download_corpus() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tgz = CACHE_DIR / "wiki.abstracts.2017.tar.gz"
    if not tgz.exists():
        print(f"[HoVer] Downloading {CORPUS_URL} (~4GB) to {tgz}...")
        urllib.request.urlretrieve(CORPUS_URL, tgz)
    print(f"[HoVer] Extracting {tgz.name} to {CACHE_DIR}...")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(path=str(CACHE_DIR))
    assert CORPUS_PATH.exists(), f"Corpus missing after extract: {CORPUS_PATH}"


def _load_corpus() -> list[str]:
    corpus = []
    with CORPUS_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            corpus.append(f"{row['title']} | {' '.join(row['text'])}")
    return corpus


def _build_index() -> None:
    print(f"[HoVer] Building BM25 index from {CORPUS_PATH}...")
    corpus = _load_corpus()
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(tokens)
    retriever.save(str(BM25_PATH))


def _ensure_retriever() -> None:
    global _retriever, _stemmer, _corpus
    if _retriever is not None:
        return
    with _init_lock:
        if _retriever is not None:
            return
        if not CORPUS_PATH.exists():
            _download_corpus()
        if not BM25_PATH.exists():
            _build_index()
        _retriever = bm25s.BM25.load(str(BM25_PATH))
        _stemmer = Stemmer.Stemmer("english")
        _corpus = _load_corpus()


def search(query: str, k: int) -> list[str]:
    _ensure_retriever()
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=_stemmer, show_progress=False)
    results, _scores = _retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    return [_corpus[doc] for doc in results[0]][:k]


# ---------- DSPy program (port of HoverMultiHop) ----------

class HoverMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
        self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
        self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")
        self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")

    def forward(self, claim):
        hop1_docs = search(claim, k=self.k)
        s1 = self.summarize1(claim=claim, passages=hop1_docs).summary
        q2 = self.create_query_hop2(claim=claim, summary_1=s1).query
        hop2_docs = search(q2, k=self.k)
        s2 = self.summarize2(claim=claim, context=s1, passages=hop2_docs).summary
        q3 = self.create_query_hop3(claim=claim, summary_1=s1, summary_2=s2).query
        hop3_docs = search(q3, k=10)
        return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)


# ---------- Data loading (port of hoverBench + Benchmark.{create_splits,trim_dataset}) ----------

def _count_unique_docs(example) -> int:
    return len(set(fact["key"] for fact in example["supporting_facts"]))


_cached_splits: tuple[list, list, list] | None = None


def _build_splits() -> tuple[list, list, list]:
    from datasets import load_dataset

    raw = load_dataset("hover", trust_remote_code=True)["train"]

    rows: list[dict] = []
    for ex in raw:
        if _count_unique_docs(ex) == 3:
            rows.append(
                dict(
                    claim=ex["claim"],
                    supporting_facts=ex["supporting_facts"],
                    label=ex["label"],
                )
            )

    rng = random.Random()
    rng.seed(0)
    rng.shuffle(rows)

    dataset = [dspy.Example(**x).with_inputs("claim") for x in rows]

    total = len(dataset)
    test_set = dataset[: int(0.4 * total)]
    val_set = dataset[int(0.4 * total) : int(0.8 * total)]
    train_set = dataset[int(0.8 * total) :]

    def trim(ds: list, size: int) -> list:
        if len(ds) <= size:
            return list(ds)
        rng = random.Random(1)
        return rng.sample(ds, size)

    return trim(train_set, 150), trim(val_set, 300), trim(test_set, 300)


def load_splits() -> tuple[list, list]:
    """Return (train=150, test=300) matching gepa-artifact's trimmed slices."""
    global _cached_splits
    if _cached_splits is None:
        _cached_splits = _build_splits()
    train, _val, test = _cached_splits
    return list(train), list(test)


# ---------- DSPy LM configuration (per-process, lazy) ----------

_lm_configured = False


def _ensure_lm() -> None:
    global _lm_configured
    if _lm_configured:
        return
    kwargs: dict = dict(
        model=TASK_MODEL,
        temperature=TASK_TEMPERATURE,
        top_p=TASK_TOP_P,
        max_tokens=TASK_MAX_TOKENS,
        timeout=TASK_TIMEOUT,
    )
    extra_body: dict = {}
    if TASK_MODEL.startswith("openrouter/") and TASK_PROVIDER_ONLY:
        extra_body["provider"] = {"only": [TASK_PROVIDER_ONLY]}
    if TASK_TOP_K:
        extra_body["top_k"] = TASK_TOP_K
    extra_body["min_p"] = TASK_MIN_P
    kwargs["extra_body"] = extra_body
    dspy.configure(lm=dspy.LM(**kwargs))
    _lm_configured = True


# ---------- Score function ----------

def _apply_bundle(program: HoverMultiHop, bundle: dict[str, str]) -> None:
    for key in ("summarize1", "create_query_hop2", "summarize2", "create_query_hop3"):
        text = bundle.get(key, "")
        if not text:
            continue
        pred = getattr(program, key).predict
        pred.signature = pred.signature.with_instructions(text)


def _gold_titles(example) -> set[str]:
    return set(
        map(
            dspy.evaluate.normalize_text,
            [doc["key"] for doc in example.supporting_facts],
        )
    )


def _found_titles(retrieved_docs: list[str]) -> set[str]:
    return set(
        map(
            dspy.evaluate.normalize_text,
            [c.split(" | ")[0] for c in retrieved_docs],
        )
    )


def _build_feedback(example, retrieved_docs: list[str], correct: bool) -> str:
    gold = _gold_titles(example)
    found = _found_titles(retrieved_docs)
    matched = sorted(gold.intersection(found))
    missing = sorted(gold.difference(found))
    if correct:
        return (
            f"Claim: {example.claim}\n"
            f"All gold supporting-fact titles retrieved: {matched}"
        )
    return (
        f"Claim: {example.claim}\n"
        f"Retrieved gold titles: {matched}\n"
        f"Missed gold titles: {missing}\n"
        "Your queries/summaries should help bridge to the missed titles."
    )


def _score_one(bundle: dict[str, str], example) -> tuple[float, list[str], str]:
    if isinstance(example, dict):
        example = dspy.Example(**example).with_inputs("claim")
    program = HoverMultiHop()
    _apply_bundle(program, bundle)
    try:
        pred = program(claim=example.claim)
        gold = _gold_titles(example)
        found = _found_titles(pred.retrieved_docs)
        match = gold.issubset(found)
        score = 1.0 if match else 0.0
        feedback = _build_feedback(example, pred.retrieved_docs, bool(match))
        return (score, pred.retrieved_docs, feedback)
    except Exception as e:
        return (0.0, [], f"Claim: {example.claim}\nPredicted: (crash: {type(e).__name__})")


def score_fn(bundle: dict[str, str], inputs: list) -> dict:
    import concurrent.futures

    _ensure_lm()

    n = len(inputs)
    per_example_scores: list[float] = [0.0] * n
    predictions: list[list[str]] = [[] for _ in range(n)]
    feedback_per_example: list[str] = [""] * n

    if n == 0:
        return {
            "score": 0.0,
            "recall": 0.0,
            "per_example_scores": [],
            "predictions": [],
            "feedback_per_example": [],
        }

    max_workers = min(TASK_MAX_WORKERS, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, bundle, ex): i for i, ex in enumerate(inputs)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            score, retrieved_docs, feedback = future.result()
            per_example_scores[i] = score
            predictions[i] = retrieved_docs
            feedback_per_example[i] = feedback

    mean = sum(per_example_scores) / len(per_example_scores)
    return {
        "score": mean,
        "recall": mean,
        "per_example_scores": per_example_scores,
        "predictions": predictions,
        "feedback_per_example": feedback_per_example,
    }

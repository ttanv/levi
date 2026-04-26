"""HotPotQA — faithful replication of gepa-artifact's hotpotQA benchmark.

Mirrors `gepa_artifact/benchmarks/hotpotQA/` line-for-line:
  - Dataset: HotpotQA fullwiki train split, sequential 40/40/20 test/val/train
    slice, then random.Random(1).sample trimmed to (150 train, 300 val, 300 test).
  - Retriever: bm25s over `wiki.abstracts.2017.jsonl` (downloaded on first run),
    BM25(k1=0.9, b=0.4), k=7, tokenizer stopwords="en" + english Stemmer.
  - Program: 4 dspy.ChainOfThought modules with the same signature strings as
    `HotpotMultiHop` (question,passages->summary ; question,summary_1->query ;
    question,context,passages->summary ; question,summary_1,summary_2->answer).
  - Scoring: `dspy.evaluate.answer_exact_match`.

Levi optimizes the 4 instruction strings plugged into each predictor's signature
at eval time via `pred.signature.with_instructions(...)`. Seed strings are
empty => DSPy uses its default auto-generated instruction, matching GEPA's
starting point.

Levi itself has no DSPy dependency; only this file does.
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
# "DO NOT use greedy decoding, as it can lead to performance degradation
# and endless repetitions."
TASK_MODEL = os.getenv("HOTPOT_TASK_MODEL", "openrouter/qwen/qwen3-8b")
TASK_PROVIDER_ONLY = os.getenv("HOTPOT_PROVIDER_ONLY", "alibaba")
TASK_TEMPERATURE = float(os.getenv("HOTPOT_TEMPERATURE", "0.6"))
TASK_TOP_P = float(os.getenv("HOTPOT_TOP_P", "0.95"))
TASK_TOP_K = int(os.getenv("HOTPOT_TOP_K", "20"))
TASK_MIN_P = float(os.getenv("HOTPOT_MIN_P", "0"))
TASK_MAX_TOKENS = int(os.getenv("HOTPOT_MAX_TOKENS", "32768"))
TASK_TIMEOUT = float(os.getenv("HOTPOT_TIMEOUT", "360"))
TASK_MAX_WORKERS = int(os.getenv("HOTPOT_MAX_WORKERS", "8"))

CACHE_DIR = Path(
    os.getenv("HOTPOT_CACHE_DIR", str(Path.home() / ".cache" / "levi" / "hotpotqa"))
)
CORPUS_URL = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
CORPUS_PATH = CACHE_DIR / "wiki.abstracts.2017.jsonl"
BM25_PATH = CACHE_DIR / "bm25s_retriever"

PROBLEM_DESCRIPTION = (
    "Optimize a 4-component prompt bundle for HotPotQA fullwiki multi-hop QA. "
    "The DSPy program: summarize1 reads hop-1 BM25 passages, create_query_hop2 "
    "writes a follow-up query, summarize2 integrates hop-2 passages, final_answer "
    "returns a short answer. Scoring is exact-match via dspy.evaluate.answer_exact_match."
)

# Seed = empty -> DSPy uses the default auto-generated instruction for each
# signature (exactly what HotpotMultiHop starts from in the artifact).
SEED_BUNDLE: dict[str, str] = {
    "summarize1": "",
    "create_query_hop2": "",
    "summarize2": "",
    "final_answer": "",
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
        print(f"[HotPot] Downloading {CORPUS_URL} (~4GB) to {tgz}...")
        urllib.request.urlretrieve(CORPUS_URL, tgz)
    print(f"[HotPot] Extracting {tgz.name} to {CACHE_DIR}...")
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
    print(f"[HotPot] Building BM25 index from {CORPUS_PATH}...")
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


def search(query: str, k: int = 7) -> list[str]:
    _ensure_retriever()
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=_stemmer, show_progress=False)
    results, _scores = _retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    return [_corpus[doc] for doc in results[0]][:k]


# ---------- DSPy program (port of HotpotMultiHop) ----------

class HotpotMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")
        self.final_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")

    def forward(self, question):
        hop1_docs = search(question, k=7)
        s1 = self.summarize1(question=question, passages=hop1_docs).summary
        q2 = self.create_query_hop2(question=question, summary_1=s1).query
        hop2_docs = search(q2, k=7)
        s2 = self.summarize2(question=question, context=s1, passages=hop2_docs).summary
        answer = self.final_answer(question=question, summary_1=s1, summary_2=s2).answer
        return dspy.Prediction(answer=answer)


# ---------- Data loading (port of HotpotQABench + Benchmark.{create_splits,trim_dataset}) ----------

_cached_splits: tuple[list, list, list] | None = None


def _build_splits() -> tuple[list, list, list]:
    from datasets import load_dataset

    raw = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)["train"]
    dataset = [dspy.Example(**x).with_inputs("question") for x in raw]

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

def _apply_bundle(program: HotpotMultiHop, bundle: dict[str, str]) -> None:
    for key in ("summarize1", "create_query_hop2", "summarize2", "final_answer"):
        text = bundle.get(key, "")
        if not text:
            continue
        pred = getattr(program, key).predict
        pred.signature = pred.signature.with_instructions(text)


def _build_feedback(example, predicted: str, correct: bool) -> str:
    gold = getattr(example, "answer", "")
    if correct:
        return f"Q: {example.question}\nPredicted: {predicted!r} (correct)"

    context = getattr(example, "context", None) or {}
    supporting = getattr(example, "supporting_facts", None) or {}
    title_to_sents: dict[str, list[str]] = {}
    if isinstance(context, dict):
        for title, sents in zip(context.get("title", []), context.get("sentences", [])):
            title_to_sents[title] = list(sents)

    support_lines: list[str] = []
    if isinstance(supporting, dict):
        for title, sid in zip(supporting.get("title", []), supporting.get("sent_id", [])):
            sents = title_to_sents.get(title, [])
            if 0 <= sid < len(sents):
                support_lines.append(f"  [{title}] {sents[sid].strip()}")

    support_text = "\n".join(support_lines) if support_lines else "  (no gold support sentences available)"
    return (
        f"Q: {example.question}\n"
        f"Predicted: {predicted!r}\n"
        f"Gold: {gold!r} (incorrect)\n"
        f"Gold supporting sentences:\n{support_text}"
    )


def _score_one(bundle: dict[str, str], example) -> tuple[float, str, str]:
    if isinstance(example, dict):
        example = dspy.Example(**example).with_inputs("question")
    # Fresh program per call: HotpotMultiHop() is just 4 ChainOfThought wrappers,
    # cheap to build, and avoids cross-thread mutation on shared predict signatures.
    program = HotpotMultiHop()
    _apply_bundle(program, bundle)
    try:
        pred = program(question=example.question)
        match = dspy.evaluate.answer_exact_match(example, pred)
        score = 1.0 if match else 0.0
        feedback = _build_feedback(example, pred.answer, bool(match))
        return (score, pred.answer, feedback)
    except Exception as e:
        return (0.0, "", f"Q: {example.question}\nPredicted: (crash: {type(e).__name__})")


def score_fn(bundle: dict[str, str], inputs: list) -> dict:
    import concurrent.futures

    _ensure_lm()

    n = len(inputs)
    per_example_scores: list[float] = [0.0] * n
    predictions: list[str] = [""] * n
    feedback_per_example: list[str] = [""] * n

    if n == 0:
        return {
            "score": 0.0,
            "em": 0.0,
            "per_example_scores": [],
            "predictions": [],
            "feedback_per_example": [],
        }

    max_workers = min(TASK_MAX_WORKERS, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one, bundle, ex): i for i, ex in enumerate(inputs)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            score, answer, feedback = future.result()
            per_example_scores[i] = score
            predictions[i] = answer
            feedback_per_example[i] = feedback

    mean_em = sum(per_example_scores) / len(per_example_scores)
    return {
        "score": mean_em,
        "em": mean_em,
        "per_example_scores": per_example_scores,
        "predictions": predictions,
        "feedback_per_example": feedback_per_example,
    }

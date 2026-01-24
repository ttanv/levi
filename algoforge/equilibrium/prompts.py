"""Prompts for Punctuated Equilibrium paradigm shift generation."""

PARADIGM_SHIFT_PROMPT = """# Algorithmic Paradigm Shift Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Current Best Solutions (From Different Behavioral Regions)

The archive has evolved through {n_evaluations} evaluations across {n_regions} behavioral regions.
Below are the best-performing solutions from each region:

{representative_solutions}

## Your Challenge: PARADIGM SHIFT

Analyze the representative solutions above and identify their core algorithmic paradigms.

Your goal is to engineer a **fundamentally different algorithmic approach** that explores untapped regions of the solution space.

### Analysis Steps:
1. **Identify current paradigms**: What algorithmic strategies do the existing solutions use? (e.g., greedy, graph-based, dynamic programming, heuristic search, brute-force with pruning, etc.)
2. **Find the gap**: What paradigms are NOT represented in the current solutions?
3. **Design a novel approach**: Synthesize a solution using a completely different conceptual framework and data structure strategy than those found in the examples

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Actively avoid the core logic, heuristics, and search patterns used in the existing solutions
3. Design a solution using a COMPLETELY DIFFERENT strategy

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, heapq, functools, etc.) and torch if needed
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders, ellipses (...), or incomplete code
- Ensure the solution handles all edge cases

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations before or after.
"""


VARIANT_GENERATION_PROMPT = """# Generate Variant of Paradigm Shift Solution

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Base Paradigm Shift Solution (Score: {base_score:.1f})
```python
{base_code}
```

## Your Task
Generate a VARIANT of the above paradigm shift solution by:
1. Keeping the core algorithmic approach intact
2. Making targeted modifications to:
   - Constants and thresholds
   - Secondary heuristics
   - Edge case handling
   - Implementation details

The variant should explore nearby regions of the solution space while preserving the novel approach.

## Output
Output ONLY the complete Python code in a ```python block.
"""

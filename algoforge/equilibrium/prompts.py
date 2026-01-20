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

Generate a **fundamentally different algorithmic approach**.

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Identify what approaches the existing solutions use
3. Design a solution using a COMPLETELY DIFFERENT strategy

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, etc.) and torch
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders or incomplete code

### COMMON ERRORS TO AVOID:
1. **Tensor dimension mismatch**: Always verify tensor shapes before operations. Use `.shape` to debug.
2. **Index out of bounds**: When indexing tensors, ensure indices are within valid range (0 to size-1).
3. **Unhandled experts**: EVERY logical expert MUST have at least 1 replica. expert_count must have NO zeros.
4. **Wrong return count**: Return EXACTLY 3 tensors: (physical_to_logical_map, logical_to_physical_map, expert_count)
5. **Sum constraint**: expert_count.sum(dim=1) MUST equal num_replicas (288) for every layer.
6. **Dtype errors**: Use torch.int64 for all output tensors.

### Why This Matters:
The evolutionary process may be stuck in local optima. Your paradigm shift could open entirely new regions of the solution space that lead to breakthrough performance.

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations.
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

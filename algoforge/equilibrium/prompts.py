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

### EPLB-Specific Constraints (MUST follow exactly):
- **Input**: weight tensor is [num_layers, 64] where 64 is num_logical_experts
- **Output 1**: physical_to_logical_map must be [num_layers, 288] with values in range [0, 63]
- **Output 2**: logical_to_physical_map must be [num_layers, 64, X] with physical slot indices or -1 for padding
- **Output 3**: expert_count must be [num_layers, 64] with expert_count.sum(dim=1) == 288 for ALL layers
- **NO zeros in expert_count**: Every logical expert (0-63) MUST have at least 1 replica
- **Index bounds**: logical expert indices are 0-63, physical slot indices are 0-287
- **Dtype**: ALL output tensors must be torch.int64

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

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

You are tasked with generating a **fundamentally different algorithmic approach**.

### Analysis Instructions:
1. Identify the CORE ASSUMPTIONS each solution makes
2. Find what INFORMATION or STRUCTURE they all IGNORE
3. Consider ENTIRELY DIFFERENT problem decompositions
4. Think about alternative DATA STRUCTURES or REPRESENTATIONS

### Requirements:
- **DO NOT** make incremental improvements to any existing solution
- **DO NOT** combine existing approaches
- **DO** design from FIRST PRINCIPLES
- **DO** explore an algorithm that uses a completely different:
  - Data representation
  - Problem decomposition
  - Optimization strategy
  - Mathematical formulation

### Why This Matters:
The evolutionary process may be stuck in local optima. Your paradigm shift could open entirely new regions of the solution space that lead to breakthrough performance.

## Output
Output ONLY the complete Python code in a ```python block.
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

"""Compare different prompts on the same evaluation set."""
import dspy
from optimize_prompts import (
    generate_mutation_examples, 
    mutation_metric,
    setup_light_model_lm,
    MutationSignature,
    MutationModule,
)
from problem import get_lazy_inputs

# Current AlgoForge prompt (from builder.py)
CURRENT_PROMPT = '''Output your improved code using SEARCH/REPLACE blocks.

FORMAT:
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

CRITICAL REQUIREMENTS:
1. The resulting code must be COMPLETE and RUNNABLE
2. Do NOT remove import statements unless replacing with different imports
3. Ensure your replacements maintain valid Python syntax
4. If adding new functionality that needs imports, add them with a separate SEARCH/REPLACE block

RULES:
1. Make SURGICAL changes - small, focused edits (5-20 lines max per block)
2. Copy the SEARCH section EXACTLY from the original (including whitespace/indentation)
3. Use multiple small SEARCH/REPLACE blocks instead of one large block
4. Start your response immediately with <<<<<<< SEARCH
5. Do NOT include any explanation or text outside the blocks
6. Do NOT use ```python code blocks

GOOD: Replace a single function or a few lines
BAD: Replace the entire file or 100+ lines at once'''

# DSPy default prompt (used as baseline in optimization)
DSPY_DEFAULT = '''Generate an improved version of code using SEARCH/REPLACE diffs.

You are given a parent solution and one inspiration solution. Mutate the parent
code to improve its score, optionally borrowing ideas from the inspiration.
Output SEARCH/REPLACE blocks to make surgical edits to the parent code.'''

# COPRO optimized prompt
COPRO_OPTIMIZED = '''You are a code enhancement expert. Given a parent solution (current code) and an inspiration solution (a reference with potential improvements), analyze both and identify opportunities to enhance the parent code—such as improving performance, readability, correctness, or efficiency—by incorporating ideas from the inspiration. 

Your task is to generate a sequence of precise, atomic SEARCH/REPLACE diffs that modify the parent code. Each diff must:
- Be syntactically correct and apply only to the parent code.
- Preserve the original logic unless an improvement is clearly justified.
- Focus on small, high-impact changes (e.g., replacing a loop with a built-in, fixing a bug, simplifying conditionals).
- Avoid introducing new dependencies or altering structure unless necessary.

Output only the SEARCH/REPLACE blocks, one per line, in the format:
```
SEARCH: <exact text to find>
REPLACE: <replacement text>
```
Do not include explanations, comments, or extra formatting.'''

# MIPROv2 best prompt (from earlier run with 37.5%)
MIPRO_BEST = '''You are tasked with generating a sequence of precise, surgical code modifications to improve a given parent code implementation, using insights from a higher-performing inspiration code. Begin by analyzing the problem description and function signature to understand the required behavior and constraints. Then, carefully compare the parent code (current implementation) with the inspiration code (a superior alternative) to identify specific, high-impact improvements—such as optimized control flow, better resource utilization, improved efficiency, or enhanced correctness—without altering the core logic or violating any strict formatting or structural requirements (e.g., tensor shapes, module interfaces, or distributed execution semantics).

Your output must be a series of SEARCH/REPLACE blocks, where each block specifies a minimal, safe, and contextually accurate edit to the parent code. Each SEARCH pattern must exactly match a fragment of the parent code, and each REPLACE replacement must integrate an improved version derived from the inspiration, while preserving syntactic correctness and compatibility with the overall system. Prioritize changes that directly contribute to improving the code's score—such as reducing latency, balancing load across GPUs/nodes, or enforcing strict tensor alignment—especially in the context of distributed Mixture-of-Experts (MoE) models with hierarchical replication.

Ensure that your edits are incremental, reversible, and do not introduce new dependencies or break existing functionality. When borrowing ideas, clearly justify the change in intent (e.g., "improve load balancing by restructuring routing logic"), but keep the output focused solely on the SEARCH/REPLACE syntax. Do not include explanations, commentary, or markdown formatting—only the raw SEARCH/REPLACE blocks in a valid, executable sequence.'''


def evaluate_prompt(prompt_name: str, instructions: str, n_examples: int = 10):
    """Evaluate a prompt on test examples."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {prompt_name}")
    print(f"{'='*60}")
    
    # Setup
    lm = setup_light_model_lm()
    dspy.configure(lm=lm)
    inputs = get_lazy_inputs()
    
    # Create module with custom instructions
    class CustomMutationSignature(dspy.Signature):
        __doc__ = instructions
        
        problem_description: str = dspy.InputField(desc="The optimization problem description")
        function_signature: str = dspy.InputField(desc="The exact function signature to implement")
        parent_code: str = dspy.InputField(desc="The parent code to mutate (v1)")
        parent_score: float = dspy.InputField(desc="Score of the parent code")
        inspiration_code: str = dspy.InputField(desc="An inspiration solution to optionally borrow ideas from (v2)")
        inspiration_score: float = dspy.InputField(desc="Score of the inspiration code")
        diff_output: str = dspy.OutputField(desc="SEARCH/REPLACE blocks to modify the parent code")
    
    class CustomModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict(CustomMutationSignature)
        
        def forward(self, **kwargs):
            return self.generate(**kwargs)
    
    module = CustomModule()
    testset = generate_mutation_examples(n_examples)
    
    # Evaluate
    scores = []
    for i, ex in enumerate(testset):
        try:
            pred = module(**ex.inputs())
            score = mutation_metric(ex, pred, inputs, debug=True)
            scores.append(score)
            print(f"  Example {i+1}: {score:.2f}")
        except Exception as e:
            print(f"  Example {i+1}: ERROR - {e}")
            scores.append(0.0)
    
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nAverage score: {avg:.1%}")
    return avg


if __name__ == "__main__":
    results = {}
    
    for name, prompt in [
        ("Current AlgoForge", CURRENT_PROMPT),
        ("DSPy Default", DSPY_DEFAULT),
        ("COPRO Optimized", COPRO_OPTIMIZED),
        ("MIPROv2 Best (37.5%)", MIPRO_BEST),
    ]:
        results[name] = evaluate_prompt(name, prompt, n_examples=10)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.1%}")

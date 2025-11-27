# AlgoForge: The method-agnostic, budget-aware framework for algorithmic discovery. 

Write your evaluator/metric once, and get access to a wide range of different algorithmic discovery methods (like AlphaEvolve, GEPA, and MCTS). The framework allows you to focus on the problem of your domain, and abstracts away the headache of dealing with the specific method or worrying about budget accounting. Whether you're using the framework for systems optimization, efficiency gains, or prompt optimization; your focus should be on the domain, not writing the scaffolding. 

The framework uses a robust, distributed, method-agnostic loop of sample -> eval -> update state primitives that form a unified framework. All other adapters are simply *instances* of this core engine. The primitives include budget accounting; thus any instance is a budget-aware by construction. This eases and standardizes research into further algorithmic discovery methods. 

### Features:
- Multiple methods through a single framework
- Built-in scaffolding (distributed engine, sandboxing) 
- Easy to add algo new algo discovery methods

### Current Methods:
- AlphaEvolve
- GEPA
- MCTS
- EoH
- Best-Of-N

## Getting Started

Install AlgoForge with `pip`

```bash
pip install algoforge
```



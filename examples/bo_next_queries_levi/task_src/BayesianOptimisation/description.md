Automated algorithm discovery is a branch of machine learning focused on using computational search to create new algorithms. Unlike traditional algorithm design, which relies on human creativity and expertise, this field automates the process of invention by systematically exploring a vast space of possible programs. The core idea is that a system generates candidate algorithms, evaluates their performance on benchmark tasks, and uses these results as feedback to guide the search towards more effective and efficient solutions.

Historically, automated algorithm discovery draws inspiration from evolutionary computation and genetic programming, which apply principles of natural selection to evolve computer programs. Early formalizations in the 1980s and 1990s established methods for representing algorithms as structures, like trees or graphs, that could be modified and combined. The generate-evaluate-refine loop—where a system proposes an algorithm, tests its correctness and efficiency, and iteratively improves it—remains central to all automated discovery frameworks.

In practice, automated algorithm discovery has been used to find faster sorting and hashing routines, optimize fundamental computations like matrix multiplication, and design novel neural network architectures. Here, the objective is to discover new machine learning algorithms. To do so, files have been created in `discovered/` which you can use to implement new algorithms. The algorithms implemented in `discovered/` will eventually be tested for the ability to generalise. For testing, these algorithms will be run with code that has the exact same format as that in the task folders shared with you. Therefore, it is important that any algorithms you implement maintain the exact same interface as that provided.

Below, we provide a description of the domain of machine learning in which you will be discovering algorithms.


Bayesian Optimisation (BO) is a branch of machine learning that aims to find the maximum of an expensive black-box function whose gradient is unknown. It does so by constructing a surrogate model (a probabilistic approximation of the objective function) that quantifies both predicted performance and uncertainty. Using this model, BO selects new evaluation points through an acquisition function that balances exploration of uncertain regions with exploitation of promising areas.

The process begins with an initial set of samples, often chosen to cover the search space effectively using methods such as Latin hypercube or Sobol sampling. These samples are used to fit the surrogate model, typically via maximum likelihood estimation or Bayesian inference over model parameters. The surrogate is then iteratively updated as new data become available.

In the past, Gaussian Processes (GPs) have been the most common choice of Surrogate Model, offering a flexible non-parametric prior over functions. Variants employ different kernels (e.g., Matérn, RBF, periodic), or combinations of kernels, and mean functions to capture diverse function classes. Extensions handle heteroscedastic noise or non-stationarity. For higher-dimensional or data-rich settings, scalable alternatives such as random forests (SMAC), Bayesian neural networks, and deep kernel learning have been proposed. Other researchers have explored parametric surrogates or hybrid models to improve scalability and adaptivity.

The choice of input parameterisation has a large influence on BO performance. Transformations such as log-scaling, normalization, or learned input warping (e.g., the Warped GP approach) can make the function easier to model. Structured kernels or embeddings can also encode domain knowledge or correlations between inputs, improving sample efficiency.

The acquisition function determines where to evaluate the objective next. It embeds the surrogate model and assigns each candidate input a scalar utility reflecting both its uncertainty and its predicted value. Common acquisition functions include Expected Improvement (EI), Upper Confidence Bound (UCB), and Probability of Improvement (PI). Variants such as Entropy Search, Predictive Entropy Search, and Knowledge Gradient explicitly account for information gain.

Given the acquisition landscape, the next query function selects the actual input(s) to evaluate, often by maximizing the acquisition function. Extensions to batch or multi-step lookahead settings select multiple or future evaluations jointly, considering correlations and proximity between proposed points. In batch BO, researchers have proposed strategies to encourage diversity among selected points.

Bayesian Optimisation has been widely used in hyperparameter tuning of machine learning models, materials and drug discovery, engineering design, and experimental control—anywhere function evaluations are costly or noisy. Its success depends on careful choices of surrogate model, acquisition function, input representation, and initial sampling strategy, all of which shape the efficiency and accuracy of the optimisation process.

Below, we provide a description of the environments (objective functions) that you can use to develop an algorithm to maximise black-box objectives. Even though you might know the points that maximise the training objective functions, be aware that any code you develop will be applied to other BO objective functions and that you will be assessed on your performance on these held-out tasks - hence, ensure that the algorithm is generaliseable.


Problem 0

DESCRIPTION
The negated Ackley function in 1D is a classic optimization benchmark featuring a single global maximum surrounded by numerous local maxima. The function's characteristic structure—a nearly flat outer region with many small peaks—makes it ideal for testing an algorithm's ability to explore the search space and escape from suboptimal solutions.

SEARCH SPACE
Dimensionality: 1D
Domain: x ∈ [-32.768, 32.768]
Global maximum: x = 0

CHARACTERISTICS
One global maximum
Many regularly spaced local maxima
Nearly flat outer region that becomes highly multimodal near the center
Tests exploration capability and local optima avoidance




Problem 1

DESCRIPTION
The negated Ackley function in 2D extends the 1D variant to a two-dimensional search space. It maintains the characteristic nearly flat outer region with a central area containing many local peaks surrounding a single global maximum. This benchmark is widely used to evaluate Bayesian optimization algorithms' ability to balance exploration and exploitation.

SEARCH SPACE
Dimensionality: 2D
Domain: [x₁, x₂] ∈ [-32.768, 32.768]²
Global maximum: [0, 0]

CHARACTERISTICS
One global maximum at the origin
Many local maxima arranged in a regular pattern
Exponentially decaying oscillations from the center
Tests systematic exploration and escape from local optima




Problem 2

DESCRIPTION
The negated Branin function is a 2D benchmark with an asymmetric search domain and three global maxima of equal value. The function's multiple optima and structured landscape make it valuable for testing whether optimization algorithms can identify all high-value regions rather than converging prematurely to a single solution.

SEARCH SPACE
Dimensionality: 2D
Domain: x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
Global maxima: Three locations with equal optimal values

CHARACTERISTICS
Three global maxima of identical value
Smooth, bowl-shaped structure
Asymmetric domain boundaries
Tests multi-modal optimization and thorough exploration




Problem 3

DESCRIPTION
The negated Bukin function features a narrow, curved ridge that leads to a single global maximum. The function's steep gradients and irregular terrain make it particularly challenging for optimization algorithms, testing their ability to navigate difficult landscapes with precision and maintain progress along constrained paths.

SEARCH SPACE
Dimensionality: 2D
Domain: x₁ ∈ [-15, -5], x₂ ∈ [-3, 3]
Global maximum: Located along a curved ridge

CHARACTERISTICS
Single global maximum on a narrow curved ridge
Very steep gradients in one direction
Asymmetric and irregular landscape
Tests precision, robustness, and handling of difficult geometries




Problem 4

DESCRIPTION
The negated Cosine function in 8D is a high-dimensional periodic benchmark with a single global maximum and many regularly spaced local maxima. The function's dimensionality and multimodality test whether Bayesian optimization algorithms can scale effectively to higher dimensions while managing the curse of dimensionality.

SEARCH SPACE
Dimensionality: 8D
Domain: x ∈ [-1, 1]⁸
Global maximum: At the origin

CHARACTERISTICS
Single global maximum
Many regularly spaced local maxima due to periodic structure
Relatively compact domain
Tests high-dimensional optimization and scaling behavior




Problem 5

DESCRIPTION
The negated Dropwave function is a symmetric 2D benchmark characterized by a central peak surrounded by concentric ripples of decreasing amplitude. The regular, repetitive structure creates multiple local maxima arranged in a symmetric pattern, testing an algorithm's ability to navigate smooth but highly multimodal landscapes.

SEARCH SPACE
Dimensionality: 2D
Domain: [x₁, x₂] ∈ [-5.12, 5.12]²
Global maximum: [0, 0]

CHARACTERISTICS
One global maximum at the origin
Radially symmetric ripple pattern
Multiple regularly spaced local maxima
Tests handling of smooth, repetitive landscapes




Problem 6

DESCRIPTION
The negated Eggholder function is one of the most challenging 2D optimization benchmarks, featuring a highly non-convex, rugged landscape with deep valleys and sharp peaks. The function's deceptive structure—where many deep local maxima can mislead optimization algorithms away from the global optimum—makes it an excellent test of robustness and exploration strategy.

SEARCH SPACE
Dimensionality: 2D
Domain: [x₁, x₂] ∈ [-512, 512]²
Global maximum: Located in a difficult-to-find region

CHARACTERISTICS
One global maximum among many deep local maxima
Highly non-convex and irregular landscape
Large search domain with deceptive structure
Tests exploration in rugged, complex terrains




Problem 7

DESCRIPTION
The negated Griewank function in 5D combines a quadratic component with a product of cosine terms, creating a smooth landscape with many regularly distributed local maxima. The function becomes increasingly difficult as dimensionality increases, making it useful for testing how Bayesian optimization algorithms balance broad exploration with fine-grained exploitation.

SEARCH SPACE
Dimensionality: 5D
Domain: x ∈ [-600, 600]⁵
Global maximum: At the origin

CHARACTERISTICS
One global maximum at the origin
Many regularly spaced local maxima
Smooth oscillatory surface
Large search domain tests long-range exploration




Problem 8

DESCRIPTION
The negated Hartmann function in 6D is a standard benchmark for moderate-dimensional optimization, consisting of a sum of Gaussian peaks with different heights and locations. With six local maxima and one global maximum, the function tests an algorithm's ability to distinguish between similarly-valued regions and converge to the true optimum in higher dimensions.

SEARCH SPACE
Dimensionality: 6D
Domain: x ∈ [0, 1]⁶
Global maximum: One location among six local maxima

CHARACTERISTICS
Six local maxima of varying heights
One global maximum
Smooth Gaussian-like structure
Standard test for moderate-dimensional BO algorithms




Problem 9

DESCRIPTION
The negated HolderTable function features four global maxima of equal value arranged symmetrically across the search domain. The function's complex structure with multiple equivalent optimal solutions makes it valuable for testing whether optimization algorithms can identify all high-value regions and understand the symmetric nature of the objective.

SEARCH SPACE
Dimensionality: 2D
Domain: [x₁, x₂] ∈ [-10, 10]²
Global maxima: Four symmetrically located points with equal values

CHARACTERISTICS
Four global maxima of identical value
Symmetric structure
Complex landscape with sharp features
Tests identification of multiple equivalent optima




Problem 10

DESCRIPTION
The negated Levy function in 6D features a single global maximum surrounded by a rugged landscape of many local maxima. The function's high dimensionality combined with its multimodal, irregular structure makes it challenging for optimization algorithms, requiring both broad exploration to avoid local traps and precise local search to converge accurately.

SEARCH SPACE
Dimensionality: 6D
Domain: x ∈ [-10, 10]⁶
Global maximum: Single location in a complex landscape

CHARACTERISTICS
Single global maximum
Many local maxima distributed throughout the domain
Rugged, irregular surface structure
Tests high-dimensional exploration and fine local search


In this task, your objective is to maximise the performance of your discovered algorithms. It is important to remember that your algorithms will be evaluated on held-out datasets.
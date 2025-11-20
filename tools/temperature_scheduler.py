"""
Temperature Scheduler for BCSR Coreset Selection

Provides adaptive temperature scheduling to promote convergence in bilevel optimization.
Temperature gradually decreases from initial to final value over max_iterations.

Supports three decay strategies:
1. Linear: Simple linear interpolation
2. Cosine: Smooth cosine annealing (recommended)
3. Exponential: Fast initial decay, slow later

Author: Claude
Date: 2024
"""

import math


class TemperatureScheduler:
    """
    Adaptive temperature scheduler for topk selection in BCSR coreset.

    Temperature controls the "softness" of topk selection:
    - High temperature (e.g., 2.0): Soft selection, more exploration
    - Low temperature (e.g., 0.5): Hard selection, faster convergence

    Args:
        initial_temp (float): Starting temperature (default: 2.0)
        final_temp (float): Ending temperature (default: 0.5)
        max_iterations (int): Total number of iterations (default: 50)
        strategy (str): Decay strategy - 'linear', 'cosine', or 'exponential' (default: 'cosine')

    Example:
        >>> scheduler = TemperatureScheduler(initial_temp=2.0, final_temp=0.5, max_iterations=50)
        >>> for i in range(50):
        ...     temp = scheduler.get_temperature()
        ...     print(f"Iteration {i}: temp={temp:.3f}")
        ...     scheduler.step()
    """

    def __init__(self, initial_temp=2.0, final_temp=0.5, max_iterations=50, strategy='cosine'):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iterations = max_iterations
        self.strategy = strategy
        self.current_iteration = 0

        # Validate parameters
        if initial_temp <= 0 or final_temp <= 0:
            raise ValueError(f"Temperatures must be positive, got initial={initial_temp}, final={final_temp}")
        if initial_temp < final_temp:
            raise ValueError(f"Initial temp ({initial_temp}) should be >= final temp ({final_temp})")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if strategy not in ['linear', 'cosine', 'exponential']:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from ['linear', 'cosine', 'exponential']")

    def get_temperature(self):
        """
        Get current temperature based on current iteration and decay strategy.

        Returns:
            float: Current temperature value
        """
        if self.current_iteration >= self.max_iterations:
            return self.final_temp

        # Calculate progress: 0.0 (start) -> 1.0 (end)
        progress = self.current_iteration / self.max_iterations

        if self.strategy == 'linear':
            # Linear interpolation: temp = start - (start - end) * progress
            temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress

        elif self.strategy == 'cosine':
            # Cosine annealing: smooth decay
            # temp = end + 0.5 * (start - end) * (1 + cos(π * progress))
            temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
                   (1 + math.cos(math.pi * progress))

        elif self.strategy == 'exponential':
            # Exponential decay: fast initial decay, slow later
            # temp = end + (start - end) * exp(-5 * progress)
            temp = self.final_temp + (self.initial_temp - self.final_temp) * \
                   math.exp(-5.0 * progress)

        return temp

    def step(self):
        """
        Advance to the next iteration.
        Call this after each outer loop iteration.
        """
        self.current_iteration += 1

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_iteration = 0

    def __repr__(self):
        return (f"TemperatureScheduler(strategy={self.strategy}, "
                f"temp={self.get_temperature():.3f}, "
                f"iter={self.current_iteration}/{self.max_iterations})")


if __name__ == "__main__":
    # Test the scheduler
    print("=" * 80)
    print("Temperature Scheduler Test")
    print("=" * 80)

    max_iter = 50
    strategies = ['linear', 'cosine', 'exponential']

    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 40)
        scheduler = TemperatureScheduler(
            initial_temp=2.0,
            final_temp=0.5,
            max_iterations=max_iter,
            strategy=strategy
        )

        # Print temperature at key iterations
        key_iters = [0, 10, 25, 40, 49]
        for i in range(max_iter):
            if i in key_iters:
                temp = scheduler.get_temperature()
                print(f"Iter {i:2d}: temp={temp:.4f}")
            scheduler.step()

    print("\n" + "=" * 80)
    print("Test completed!")

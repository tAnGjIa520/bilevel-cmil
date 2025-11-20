"""
Learning Rate Scheduler for BCSR Coreset Selection

Provides adaptive learning rate scheduling to promote convergence in bilevel optimization.
Learning rate gradually decreases from initial to final value over max_iterations.

Supports three decay strategies:
1. Linear: Simple linear interpolation
2. Cosine: Smooth cosine annealing (recommended)
3. Exponential: Fast initial decay, slow later

Author: Claude
Date: 2024
"""

import math


class LearningRateScheduler:
    """
    Adaptive learning rate scheduler for weight optimizer in BCSR coreset.

    Learning rate controls the step size for weight updates:
    - High learning rate (e.g., 0.001): Fast updates, may overshoot
    - Low learning rate (e.g., 0.0001): Slow updates, more stable convergence

    Args:
        initial_lr (float): Starting learning rate (default: 0.001)
        final_lr (float): Ending learning rate (default: 0.0001)
        max_iterations (int): Total number of iterations (default: 50)
        strategy (str): Decay strategy - 'linear', 'cosine', or 'exponential' (default: 'cosine')

    Example:
        >>> scheduler = LearningRateScheduler(initial_lr=0.001, final_lr=0.0001, max_iterations=50)
        >>> for i in range(50):
        ...     lr = scheduler.get_lr()
        ...     print(f"Iteration {i}: lr={lr:.6f}")
        ...     scheduler.step()
    """

    def __init__(self, initial_lr=0.001, final_lr=0.0001, max_iterations=50, strategy='cosine'):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.max_iterations = max_iterations
        self.strategy = strategy
        self.current_iteration = 0

        # Validate parameters
        if initial_lr <= 0 or final_lr <= 0:
            raise ValueError(f"Learning rates must be positive, got initial={initial_lr}, final={final_lr}")
        if initial_lr < final_lr:
            raise ValueError(f"Initial lr ({initial_lr}) should be >= final lr ({final_lr})")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if strategy not in ['linear', 'cosine', 'exponential']:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from ['linear', 'cosine', 'exponential']")

    def get_lr(self):
        """
        Get current learning rate based on current iteration and decay strategy.

        Returns:
            float: Current learning rate value
        """
        if self.current_iteration >= self.max_iterations:
            return self.final_lr

        # Calculate progress: 0.0 (start) -> 1.0 (end)
        progress = self.current_iteration / self.max_iterations

        if self.strategy == 'linear':
            # Linear interpolation: lr = start - (start - end) * progress
            lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress

        elif self.strategy == 'cosine':
            # Cosine annealing: smooth decay
            # lr = end + 0.5 * (start - end) * (1 + cos(π * progress))
            lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * \
                   (1 + math.cos(math.pi * progress))

        elif self.strategy == 'exponential':
            # Exponential decay: fast initial decay, slow later
            # lr = end + (start - end) * exp(-5 * progress)
            lr = self.final_lr + (self.initial_lr - self.final_lr) * \
                   math.exp(-5.0 * progress)

        return lr

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
        return (f"LearningRateScheduler(strategy={self.strategy}, "
                f"lr={self.get_lr():.6f}, "
                f"iter={self.current_iteration}/{self.max_iterations})")


if __name__ == "__main__":
    # Test the scheduler
    print("=" * 80)
    print("Learning Rate Scheduler Test")
    print("=" * 80)

    max_iter = 50
    strategies = ['linear', 'cosine', 'exponential']

    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 40)
        scheduler = LearningRateScheduler(
            initial_lr=0.001,
            final_lr=0.0001,
            max_iterations=max_iter,
            strategy=strategy
        )

        # Print learning rate at key iterations
        key_iters = [0, 10, 25, 40, 49]
        for i in range(max_iter):
            if i in key_iters:
                lr = scheduler.get_lr()
                print(f"Iter {i:2d}: lr={lr:.6f}")
            scheduler.step()

    print("\n" + "=" * 80)
    print("Test completed!")

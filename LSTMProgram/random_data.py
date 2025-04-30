import random

# Set seed for reproducibility
random.seed(42)

all_numbers = set(range(1, 21))
combinations = []

# We want 10 non-overlapping 4-number combinations from 20 total numbers
numbers = list(range(1, 21))
random.shuffle(numbers)

# Create 10 non-overlapping groups of 4 from shuffled numbers
for i in range(0, 20, 2):
    test_group = numbers[i:i+4]
    train_group = list(set(numbers) - set(test_group))
    combinations.append((sorted(train_group), sorted(test_group)))


print(combinations)
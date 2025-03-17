import json
import random

# Function to load and create a balanced dataset
def create_balanced_dataset(full_data, reviews_per_star):
    # Group the reviews by their star ratings
    star_groups = {1: [], 2: [], 3: [], 4: [], 5: []}

    # Group reviews by their star rating
    for review in full_data:
        star = review['stars']
        if star in star_groups:
            star_groups[star].append(review)

    # Create a balanced dataset by sampling an equal number of reviews from each star group
    balanced_data = []
    for star in range(1, 6):
        # Sample the specified number of reviews for each star rating
        sampled_reviews = random.sample(star_groups[star], reviews_per_star)
        balanced_data.extend(sampled_reviews)

    return balanced_data

# Load the full training data
full_train_data_path = 'training.json'  # Update this path
with open(full_train_data_path, 'r') as file:
    full_train_data = json.load(file)

# Create the balanced dataset (e.g., 1000 reviews per star)
balanced_data = create_balanced_dataset(full_train_data, 250)

# Save the balanced dataset to a new file
balanced_data_path = 'tiny_training.json'  # Path to save the balanced data
with open(balanced_data_path, 'w') as file:
    json.dump(balanced_data, file)

print(f"Balanced dataset created with {len(balanced_data)} reviews.")
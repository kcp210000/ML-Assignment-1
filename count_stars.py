import json

# Path to your JSON file (change this to the correct path)
file_path = 'tiny_training.json'

# Open and load the JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize a dictionary to count the stars
star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# Iterate over each review in the dataset
for review in data:
    # Get the star rating for the review (assumes it's a number in the "stars" field)
    stars = review['stars']
    
    # Check if the star rating is valid (1-5)
    if stars in star_counts:
        star_counts[stars] += 1

# Print out the count of each star rating
print("Star counts:")
for star, count in star_counts.items():
    print(f"{star} stars: {count} reviews")
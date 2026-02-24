#import required packages
import numpy as np
import pandas as pd


def generate_candidate_data(n_samples=800, random_state=42):
    """
    Generates a synthetic dataset for candidate performance prediction.

    Parameters:
    n_samples (int): Number of candidates to generate.
    random_state (int): Seed for reproducibility.

    Returns:
    pd.DataFrame: Simulated candidate dataset.
    """

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # ------------------------------
    # 1. Generate Candidate Features
    # ------------------------------

    # Education levels (categorical feature)
    education_levels = ["Diploma", "Bachelor", "Master"]

    # Randomly assign education levels with given probabilities
    education = np.random.choice(
        education_levels,
        size=n_samples,
        p=[0.3, 0.5, 0.2]  # 30% Diploma, 50% Bachelor, 20% Master
    )

    # Map education level to numeric bonus
    education_bonus_map = {
        "Diploma": 0,
        "Bachelor": 5,
        "Master": 10
    }

    # Convert to numeric bonus array
    education_bonus = np.array([education_bonus_map[level] for level in education])


    # Years of experience (integer between 0 and 15)
    years_experience = np.random.randint(0, 16, size=n_samples)

    # Technical test scores (normally distributed around 70)
    technical_test_score = np.random.normal(
        loc=70,   # mean
        scale=10, # standard deviation
        size=n_samples
    ).clip(40, 100)  # keep scores between 40 and 100

    # Interview scores (normally distributed around 65)
    interview_score = np.random.normal(
        loc=65,
        scale=12,
        size=n_samples
    ).clip(35, 100)

    # Personality assessment scores
    personality_score = np.random.normal(
        loc=75,
        scale=8,
        size=n_samples
    ).clip(50, 100)

    # Previous performance rating (1–5 scale)
    previous_rating = np.random.normal(
        loc=3.5,
        scale=0.8,
        size=n_samples
    ).clip(1, 5)

    # ----------------------------------
    # 2. Create Hidden Performance Logic
    # ----------------------------------

    # Weighted formula to simulate realistic performance influence
    performance_score = (
        0.3 * technical_test_score +
        0.25 * interview_score +
        0.2 * personality_score +
        2 * years_experience +
        5 * previous_rating +
        education_bonus
    )

    print("Min score:", performance_score.min())
    print("Max score:", performance_score.max())
    # Convert continuous performance score into categories
    performance_label = pd.cut(
        performance_score,
        bins=[0, 85, 105, np.inf],
        labels=["Low", "Average", "High"]
    )

    # -----------------------------
    # 3. Create Final DataFrame
    # -----------------------------

    data = pd.DataFrame({
        "education_level": education,
        "years_experience": years_experience,
        "technical_test_score": technical_test_score.round(1),
        "interview_score": interview_score.round(1),
        "personality_score": personality_score.round(1),
        "previous_rating": previous_rating.round(1),
        "performance_score": performance_score,
        "performance_label": performance_label
    })

    return data


# ----------------------------------
# Run script directly (not imported)
# ----------------------------------
if __name__ == "__main__":

    # Generate dataset
    df = generate_candidate_data()

    # Save to CSV inside data folder
    df.to_csv("./data/simulated_candidates.csv", index=False)

    print("Dataset generated and saved successfully.")


import pandas as pd
import random

# Number of sample rows
n_samples = 50

data = {
    "Age": [random.randint(18, 70) for _ in range(n_samples)],
    "TypeofContact": [random.choice(["Self Enquiry", "Company Invited"]) for _ in range(n_samples)],
    "CityTier": [random.choice([1, 2, 3]) for _ in range(n_samples)],
    "DurationOfPitch": [random.randint(1, 60) for _ in range(n_samples)],
    "Occupation": [random.choice(["Free Lancer", "Salaried", "Small Business", "Large Business"]) for _ in range(n_samples)],
    "Gender": [random.choice(["Male", "Female"]) for _ in range(n_samples)],
    "NumberOfPersonVisiting": [random.randint(1, 10) for _ in range(n_samples)],
    "NumberOfFollowups": [random.randint(0, 5) for _ in range(n_samples)],
    "PreferredPropertyStar": [random.choice([1, 2, 3, 4, 5]) for _ in range(n_samples)],
    "MaritalStatus": [random.choice(["Single", "Unmarried", "Divorced", "Married"]) for _ in range(n_samples)],
    "NumberOfTrips": [random.randint(0, 10) for _ in range(n_samples)],
    "Passport": [random.choice([0, 1]) for _ in range(n_samples)],
    "PitchSatisfactionScore": [random.randint(1, 5) for _ in range(n_samples)],
    "OwnCar": [random.choice([0, 1]) for _ in range(n_samples)],
    "NumberOfChildrenVisiting": [random.randint(0, 5) for _ in range(n_samples)],
    "Designation": [random.choice(["Executive", "Manager", "Senior Manager", "AVP", "VP"]) for _ in range(n_samples)],
    "MonthlyIncome": [random.randint(15000, 200000) for _ in range(n_samples)]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("sample_bulk_test_data.csv", index=False)
print("testing.csv")

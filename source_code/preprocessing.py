import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    df.drop(columns=["Student_ID"], inplace=True)

    df["Addicted"] = df["Addicted_Score"].apply(lambda x: 1 if x >= 6 else 0)
    df.drop(columns=["Addicted_Score"], inplace=True)

    categorical_cols = [
        "Gender",
        "Academic_Level",
        "Country",
        "Most_Used_Platform",
        "Affects_Academic_Performance",
        "Relationship_Status",
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    numerical_cols = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media",
    ]

    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=["Addicted"]).values
    y = df["Addicted"].values

    return train_test_split(X, y, test_size=0.2, random_state=42)

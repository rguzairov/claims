import pandas as pd


def read_claims(path="app/data/claims_sample_data__cleaned.csv"):
    claims = pd.read_csv(path)
    claims = claims[claims["MONTH_DATE"] != "2020-07-01"]
    claims["MONTH_DATE"] = pd.to_datetime(claims["MONTH_DATE"]).dt.date
    return claims
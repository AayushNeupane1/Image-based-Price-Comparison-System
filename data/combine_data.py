import pandas as pd

df1 = pd.read_csv("./raw/Personal_Care.csv")
df2 = pd.read_csv("./raw/Food_Beverages.csv")
df3 = pd.read_csv("./raw/Cosmetics.csv")
df4 = pd.read_csv("./raw/Household.csv")
df5 = pd.read_csv("./raw/Sports.csv")
df6 = pd.read_csv("./raw/Personal_Care1.csv")
df7 = pd.read_csv("./raw/Food_Beverages1.csv")
df8 = pd.read_csv("./raw/Cosmetics1.csv")
df9 = pd.read_csv("./raw/Household1.csv")
df10 = pd.read_csv("./raw/Sports1.csv")

combined = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)
combined.to_csv("products_all.csv", index=False)

print(f"Total rows: {len(combined)}")
print(combined.head())
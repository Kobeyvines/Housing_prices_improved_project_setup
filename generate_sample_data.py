"""Script to generate sample House Prices dataset for testing."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(n_samples=100):
    """Generate synthetic House Prices dataset.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic house prices data
    """
    np.random.seed(42)

    data = {
        "Id": range(1, n_samples + 1),
        "MSSubClass": np.random.choice([20, 30, 40, 50, 60, 70, 80, 90], n_samples),
        "MSZoning": np.random.choice(
            ["A", "C", "FV", "I", "RH", "RL", "RP", "RM"], n_samples
        ),
        "LotFrontage": np.random.normal(70, 20, n_samples).astype(int),
        "LotArea": np.random.exponential(10000, n_samples).astype(int),
        "Street": np.random.choice(["Grvl", "Pave"], n_samples),
        "Alley": np.random.choice(["Grvl", "Pave", "NA"], n_samples),
        "LotShape": np.random.choice(["Reg", "IR1", "IR2", "IR3"], n_samples),
        "LandContour": np.random.choice(["Lvl", "Bnk", "HLS", "Low"], n_samples),
        "Utilities": np.random.choice(
            ["AllPub", "NoSewr", "NoSewer", "ELO"], n_samples
        ),
        "LotConfig": np.random.choice(
            ["Inside", "Corner", "CulDSac", "FR2", "FR3"], n_samples
        ),
        "LandSlope": np.random.choice(["Gtl", "Mod", "Sev"], n_samples),
        "Neighborhood": np.random.choice(
            [
                "Blmngtn",
                "Blueste",
                "BrDale",
                "BrkSide",
                "ClearCr",
                "CollgCr",
                "Crawfor",
                "Edwards",
                "Gilbert",
                "IDOTRR",
                "MeadowV",
                "Mitchel",
                "Names",
                "NoRidge",
                "NPkVill",
                "NridgHt",
                "NWAmes",
                "OldTown",
                "SWISU",
                "Sawyer",
                "SawyerW",
                "Somerst",
                "StoneBr",
                "Timber",
                "Veenker",
            ],
            n_samples,
        ),
        "Condition1": np.random.choice(
            ["Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRNe", "RRNn"],
            n_samples,
        ),
        "Condition2": np.random.choice(
            ["Artery", "Feedr", "Norm", "PosA", "PosN", "RRAe", "RRNe", "RRNn"],
            n_samples,
        ),
        "BldgType": np.random.choice(
            ["1Fam", "2fmCon", "Duplex", "TwnhsE", "TwnhsI"], n_samples
        ),
        "HouseStyle": np.random.choice(
            [
                "1Story",
                "1.5Fin",
                "1.5Unf",
                "2Story",
                "2.5Fin",
                "2.5Unf",
                "SFoyer",
                "SLvl",
            ],
            n_samples,
        ),
        "OverallQual": np.random.randint(1, 10, n_samples),
        "OverallCond": np.random.randint(1, 10, n_samples),
        "YearBuilt": np.random.randint(1872, 2010, n_samples),
        "YearRemodAdd": np.random.randint(1872, 2010, n_samples),
        "RoofStyle": np.random.choice(
            ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"], n_samples
        ),
        "RoofMatl": np.random.choice(
            [
                "ClyTile",
                "CompShg",
                "Membran",
                "Metal",
                "Roll",
                "Tar&Grv",
                "WdShake",
                "WdShngl",
            ],
            n_samples,
        ),
        "Exterior1st": np.random.choice(
            [
                "AsbShng",
                "AsphShn",
                "BrkComm",
                "BrkFace",
                "CBlock",
                "CemBd",
                "HdBoard",
                "ImStucc",
                "MetalSd",
                "Other",
                "Plywood",
                "PreCast",
                "Stone",
                "Stucco",
                "VinylSd",
                "Wd Sdng",
                "WdShing",
            ],
            n_samples,
        ),
        "Exterior2nd": np.random.choice(
            [
                "AsbShng",
                "AsphShn",
                "BrkComm",
                "BrkFace",
                "CBlock",
                "CemBd",
                "HdBoard",
                "ImStucc",
                "MetalSd",
                "Other",
                "Plywood",
                "PreCast",
                "Stone",
                "Stucco",
                "VinylSd",
                "Wd Sdng",
                "WdShing",
            ],
            n_samples,
        ),
        "MasVnrType": np.random.choice(
            ["BrkCmn", "BrkFace", "CBlock", "None", "Stone"], n_samples
        ),
        "MasVnrArea": np.random.exponential(100, n_samples).astype(int),
        "ExterQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], n_samples),
        "ExterCond": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po"], n_samples),
        "Foundation": np.random.choice(
            ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"], n_samples
        ),
        "BsmtQual": np.random.choice(["Ex", "Gd", "TA", "Fa", "NA"], n_samples),
        "BsmtCond": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po", "NA"], n_samples),
        "BsmtExposure": np.random.choice(["Gd", "Av", "Mn", "No", "NA"], n_samples),
        "BsmtFinType1": np.random.choice(
            ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"], n_samples
        ),
        "BsmtFinSF1": np.random.exponential(1000, n_samples).astype(int),
        "BsmtFinType2": np.random.choice(
            ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"], n_samples
        ),
        "BsmtFinSF2": np.random.exponential(500, n_samples).astype(int),
        "BsmtUnfSF": np.random.exponential(1000, n_samples).astype(int),
        "TotalBsmtSF": np.random.exponential(2000, n_samples).astype(int),
        "Heating": np.random.choice(
            ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"], n_samples
        ),
        "HeatingQC": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po"], n_samples),
        "CentralAir": np.random.choice(["Y", "N"], n_samples),
        "Electrical": np.random.choice(
            ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"], n_samples
        ),
        "1stFlrSF": np.random.normal(1100, 400, n_samples).astype(int),
        "2ndFlrSF": np.random.exponential(800, n_samples).astype(int),
        "LowQualFinSF": np.random.exponential(200, n_samples).astype(int),
        "GrLivArea": np.random.normal(1500, 500, n_samples).astype(int),
        "BsmtFullBath": np.random.randint(0, 4, n_samples),
        "BsmtHalfBath": np.random.randint(0, 3, n_samples),
        "FullBath": np.random.randint(0, 5, n_samples),
        "HalfBath": np.random.randint(0, 3, n_samples),
        "BedroomAbvGr": np.random.randint(1, 8, n_samples),
        "KitchenAbvGr": np.random.randint(1, 4, n_samples),
        "KitchenQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], n_samples),
        "TotRmsAbvGrd": np.random.randint(2, 15, n_samples),
        "Functional": np.random.choice(
            ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev"], n_samples
        ),
        "Fireplaces": np.random.randint(0, 5, n_samples),
        "FireplaceQu": np.random.choice(
            ["Ex", "Gd", "TA", "Fa", "Po", "NA"], n_samples
        ),
        "GarageType": np.random.choice(
            ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"],
            n_samples,
        ),
        "GarageYrBlt": np.random.randint(1872, 2010, n_samples),
        "GarageFinish": np.random.choice(["Fin", "RFn", "Unf", "NA"], n_samples),
        "GarageCars": np.random.randint(0, 6, n_samples),
        "GarageArea": np.random.exponential(500, n_samples).astype(int),
        "GarageQual": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po", "NA"], n_samples),
        "GarageCond": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po", "NA"], n_samples),
        "PavedDrive": np.random.choice(["Y", "P", "N"], n_samples),
        "WoodDeckSF": np.random.exponential(200, n_samples).astype(int),
        "OpenPorchSF": np.random.exponential(150, n_samples).astype(int),
        "EnclosedPorch": np.random.exponential(100, n_samples).astype(int),
        "3SsnPorch": np.random.exponential(50, n_samples).astype(int),
        "ScreenPorch": np.random.exponential(100, n_samples).astype(int),
        "PoolArea": np.random.exponential(50, n_samples).astype(int),
        "PoolQC": np.random.choice(["Ex", "Gd", "TA", "Fa", "NA"], n_samples),
        "Fence": np.random.choice(["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"], n_samples),
        "MiscFeature": np.random.choice(
            ["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"], n_samples
        ),
        "MiscVal": np.random.exponential(500, n_samples).astype(int),
        "MoSold": np.random.randint(1, 13, n_samples),
        "YrSold": np.random.randint(2006, 2011, n_samples),
        "SaleType": np.random.choice(
            ["WD", "CWD", "VSD", "New", "COO", "Con", "ConLw", "ConLI", "ConLD", "Oth"],
            n_samples,
        ),
        "SaleCondition": np.random.choice(
            ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"], n_samples
        ),
        "SalePrice": np.random.normal(180000, 80000, n_samples).astype(int),
    }

    df = pd.DataFrame(data)

    # Ensure SalePrice is positive
    df["SalePrice"] = df["SalePrice"].clip(lower=50000)

    return df


def main():
    """Generate and save sample data."""
    print("Generating sample House Prices dataset...")

    # Create data directory if needed
    Path("data/01-raw").mkdir(parents=True, exist_ok=True)

    # Generate train data
    train_df = generate_sample_data(n_samples=1460)
    train_df.to_csv("data/01-raw/train.csv", index=False)
    print(f"✓ Saved train data: data/01-raw/train.csv ({len(train_df)} rows)")

    # Generate test data (without SalePrice)
    test_df = generate_sample_data(n_samples=1459)
    test_df = test_df.drop(columns=["SalePrice"])
    test_df["Id"] = range(1461, 1461 + len(test_df))
    test_df.to_csv("data/01-raw/test.csv", index=False)
    print(f"✓ Saved test data: data/01-raw/test.csv ({len(test_df)} rows)")

    # Create data_description.txt
    description = """FEATURES DESCRIPTION
=====================

Id: Unique identifier for each property

MSSubClass: Identifies the type of dwelling involved in the sale
- 20: 1-STORY 1946 & NEWER ALL STYLES
- 30: 1-STORY 1945 & OLDER
- 40: 1-STORY W/FINISHED ATTIC ALL AGES
- ... (more classes)

... (more features descriptions)

SalePrice: The property's sale price in dollars. This is the target variable.
"""

    with open("data/01-raw/data_description.txt", "w") as f:
        f.write(description)
    print("✓ Saved data description: data/01-raw/data_description.txt")


if __name__ == "__main__":
    main()

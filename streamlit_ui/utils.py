import pandas as pd

def transform_for_model(df_row: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """
    Transforms a single row DataFrame to match the one-hot encoded format expected by the model.
    Fills missing dummy columns with 0 and reorders as per model_columns.
    """
    df_encoded = pd.get_dummies(df_row)

    # Add missing columns
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns
    df_encoded = df_encoded[model_columns]

    return df_encoded

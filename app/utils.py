import pandas as pd

# Function to save the result as an Excel file
def save_to_excel(df, filename="output.xlsx"):
    """
    Save the DataFrame to an Excel file.

    Parameters:
    df (pandas DataFrame): DataFrame containing the results to be saved.
    filename (str): The output Excel file name.

    Returns:
    str: The filename of the saved Excel file.
    """
    df.to_excel(filename, index=False)
    return filename
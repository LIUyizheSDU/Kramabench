"""
What is the total number of money befrauded when summed over all payment methods. Give an integer number in millions of dollars.
 -- easy
 -- number
 -- 2024_Fraud_Reports_by_Payment_Method.csv
"""
from data_utils import read_clean_numeric_csv


df = read_clean_numeric_csv("../input/raw/CSVs/2024_CSN_Fraud_Reports_by_Payment_Method.csv")

total_loss_millions = int(df[" Total $ Loss"].sum() / 1_000_000)
print(total_loss_millions)



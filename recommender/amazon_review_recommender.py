import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

######################################################################################################
# TASK 1: Calculate Average Rating Based on Current Comments and Compare with Existing Average Rating.
######################################################################################################

# In the shared data set, users gave points and comments to a product.
# Our aim in this task is to evaluate the scores given by weighting them by date.
# It is necessary to compare the first average score with the weighted score according to the date to be obtained.

######################################################################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
######################################################################################################
def read_data():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(current_dir, '..', 'datasets', 'amazon_review.csv')
    print(file_path)
    #df = pd.read_csv('datasets/amazon_review.csv')
    df = pd.read_csv(file_path)
    return df

def missing_values_analysis(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=True)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

######################################################################################################
# Step 2: Calculate the Weighted Average of Score by Date.
######################################################################################################
def weighted_rating(df, m, C):
    """
    It is used to calculate a weighted score for a product. This score is based on both the product's
    popularity and users' average rating for the product.

    Inputs:

    df: A DataFrame containing the properties of the products
    m: Minimum number of votes
    C: Average score

    R = df['overall'] -> Gets the column containing the average score for each product.
    v = df['total_vote'] -> Gets the column containing the total number of votes for each product.

    The function calculates a weighted score for each product in the df DataFrame. For each product,
    the function determines the total number of votes (v) and the average score (R). It then calculates
    the weighted score using the formula.

    The functioning of the function is as follows:

    m = 1: Determines the minimum number of votes. The function necessarily takes this value, but a specific
    value is assigned within the function.

    if v == 0: The function returns 0 if the total number of votes is zero.

    else: If the total number of votes is non-zero, calculate the weighted score using the following formula:

    (v / (v + m) * R): A score based on the popularity of the product
    (m / (m + v) * C): A score based on the average rating of the product

    These two scores are added together and the weighted score is calculated.

    Finally, the calculated weighted score is returned by the function.
    This function calculates a weighted score based on the popularity and average score of the product,
    and using these scores you can rank the products.
    """
    v = df['total_vote']
    R = df['overall']
    m = 1
    if v == 0:
        return 0
    else:
        return (v / (v + m) * R) + (m / (m + v) * C)

def prep_data(df):
    df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%Y-%m-%d')
    # df['reviewTime'][0]: Timestamp('2014-07-23 00:00:00')
    df['weighted_rating'] = df.apply(lambda x: weighted_rating(x, 1, product_rating), axis=1)
    return df

# Calculation of all weighted average ratings on a year-month basis with matplotlib
def plot_weighted_average_ratings(df):
    grouped = df.groupby(pd.Grouper(key='reviewTime', freq='M')).agg({'weighted_rating': np.mean})
    plt.plot(grouped.index, grouped['weighted_rating'])
    plt.title('Monthly Weighted Average Ratings')
    plt.xlabel('Time')
    plt.ylabel('Weighted Average Rating')
    plt.show()
    
######################################################################################################
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
######################################################################################################
# Note:
# total_vote is the total number of up-downs given to a comment. Up means helpful.
# There is no helpful_no variable in the data set, it must be generated over existing variables.

######################################################################################################
# Step 1. Generate the helpful_no variable
######################################################################################################
def gen_helpful_no(df):
    df['helpful_no'] = df['total_vote'] - df['helpful_yes']
    return df

######################################################################################################
# Step 2. Calculate "score_pos_neg_diff", "score_average_rating" and "wilson_lower_bound" scores 
######################################################################################################
def score_pos_neg_diff(positive, negative):
    return positive - negative

def score_average_rating(up_vote, down_vote):
    total_vote = up_vote + down_vote
    if total_vote == 0:
        return 0
    else:
        score = up_vote / (up_vote + down_vote)
        return score

def wilson_lower_bound(up_vote, down_vote, confidence=0.95):
    total_vote = up_vote + down_vote
    if total_vote == 0:
        return 0
    else:
        z = norm.ppf(1 - (1 - confidence) / 2)
        phat = 1.0 * up_vote / total_vote
        score = (phat + z * z / (2 * total_vote) - z * np.sqrt((phat * (1 - phat) + z * z / (4 * total_vote))
                                                               / total_vote)) / (1 + z * z / total_vote)
        return score

if __name__=='__main__':
    ##################### 1. READ DATA #############################
    df = read_data()
    print("1. dataframe is read succesfully")

    ##################### 2. MISSING VALUE ANALYSIS #############################
    print(missing_values_analysis(df))
    print("2.missing values analysis is done")
    product_rating = df['overall'].mean()
    print("overall product_rating is: ", product_rating)

    ##################### 3. PREP DATA #############################
    df = prep_data(df)
    print("2. data prep is done")
    print("weighted_rating is calculated: ", df['weighted_rating'].mean())
    plot_weighted_average_ratings(df)

    ##################### 4. GENERATE HELPFUL_NO #############################
    df = gen_helpful_no(df)
    print("3. helpful_no is generated")

    ##################### 5. POS NEG DIFFERENCE #############################
    df['score_pos_neg_diff'] = df.apply(lambda x: score_pos_neg_diff(x['helpful_yes'], x['helpful_no']), axis=1)
    print("4. score_pos_neg_diff is calculated")

    ##################### 6. SCORE AVERAGE RATING #############################
    df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']), axis=1)
    print("5. score_average_rating is calculated")

    ##################### 7. WILSON LOWER BOUND #############################
    df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)
    print("6. wilson_lower_bound is calculated")

    ##################### 7. SORT WILSON LOWER BOUND DESC ORDER #############################
    #Identify 20 Comments and Interpret Results.
    print(df.sort_values("wilson_lower_bound", ascending=False).head(20))
    df.describe().T
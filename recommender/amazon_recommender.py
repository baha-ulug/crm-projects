import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import os
import importlib
warnings.filterwarnings('ignore')
sns.set_theme(color_codes=True)


def load_df():
    #load the dataframe and set column name
    df=pd.read_csv('amazon-product-reviews/ratings_Electronics.csv',
                   names=['userId', 'productId','rating','timestamp'])
    df=df.sample(n=1564896,ignore_index=True)
    return df

electronics_data = load_df()

def df_info(df):
    print(df.info())

    #observe missing values
    print(df.isnull().sum())

    #observe duplicate records
    df[df.duplicated()].shape[0]

df_info(electronics_data)

def prep_df(df):
    #drop timestamp column
    df.drop('timestamp',axis=1,inplace=True)
    return df

electronics_data = prep_df(electronics_data)

def plot_df(df):
    plt.figure(figsize=(8,4))
    sns.countplot(x='rating',data=df)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    print('Total rating : ',df.shape[0])
    print('Total unique users : ',df['userId'].unique().shape[0])
    print('Total unique products : ',df['productId'].unique().shape[0])

plot_df(electronics_data)

##################################################
# Popularity Based Recommendation
##################################################
"""
Popülariteye dayalı öneri sistemi trend ile çalışır. 
Temel olarak şu anda trend olan öğeleri kullanır. 
Örneğin, genellikle her yeni kullanıcı tarafından satın alınan herhangi bir ürün varsa, 
o ürünü yeni kaydolan kullanıcıya önerme olasılığı vardır.

Popülariteye dayalı öneri sistemindeki sorun, 
kişiselleştirmenin bu yöntemle mümkün olmamasıdır, 
yani kullanıcının davranışını bilseniz bile öğeleri buna göre öneremezsiniz.
"""
data=electronics_data.groupby('productId').filter(lambda x:x['rating'].count()>=50)
no_of_rating_per_product=data.groupby('productId')['rating'].count().sort_values(ascending=False)
no_of_rating_per_product.head()

#top 20 product
no_of_rating_per_product.head(20).plot(kind='bar')
plt.xlabel('Product ID')
plt.ylabel('num of rating')
plt.title('top 20 procduct')
plt.show()


#average rating product
mean_rating_product_count=pd.DataFrame(data.groupby('productId')['rating'].mean())
mean_rating_product_count['rating_counts'] = pd.DataFrame(data.groupby('productId')['rating'].count())
mean_rating_product_count.head()

#plot the rating distribution of average rating product
plt.hist(mean_rating_product_count['rating'],bins=100)
plt.title('Mean Rating distribution')
plt.show()

#check the skewness of the mean rating data
mean_rating_product_count['rating'].skew()
#it is highly negative skewed

#highest mean rating product
mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].max()]

#min mean rating product
print('min average rating product : ',mean_rating_product_count['rating_counts'].min())
print('total min average rating products : ',mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].min()].shape[0])

#plot the rating count of mean_rating_product_count
plt.hist(mean_rating_product_count['rating_counts'],bins=100)
plt.title('rating count distribution')
plt.show()

#joint plot of rating and rating counts
sns.jointplot(x='rating',y='rating_counts',data=mean_rating_product_count)
plt.title('Joint Plot of rating and rating counts')
plt.tight_layout()
plt.show()

plt.scatter(x=mean_rating_product_count['rating'],y=mean_rating_product_count['rating_counts'])
plt.show()

print('Correlation between Rating and Rating Counts is : {} '.format(mean_rating_product_count['rating'].corr(mean_rating_product_count['rating_counts'])))

##################################################
# Collaborative filtering Recommendation
##################################################
"""
Collaborative filtering, tavsiye sistemleri için yaygın olarak kullanılır. 
Bu teknikler, bir kullanıcı-öğe ilişkilendirme matrisinin eksik girişlerini doldurmayı amaçlar. 
Collaborative filtering (CF) yaklaşımını kullanacağız. 
CF, en iyi tavsiyelerin benzer zevklere sahip insanlardan geldiği fikrine dayanmaktadır. 
Başka bir deyişle, birinin bir öğeyi nasıl derecelendireceğini tahmin etmek için 
benzer düşünen kişilerin tarihsel öğe derecelendirmelerini kullanır. 
Collaborative filtering'in, genellikle bellek tabanlı ve model tabanlı yaklaşımlar olarak adlandırılan 
iki alt kategorisi vardır.
"""
#import surprise library for collebrative filtering
from surprise import KNNWithMeans, Dataset, accuracy, Reader
from surprise.model_selection import train_test_split

def load_data(data):
    # Load dataset
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data, reader)
    return surprise_data

def split_data(surprise_data, test_size=0.3, random_state=42):
    # Split dataset into train and test sets
    trainset, testset = train_test_split(surprise_data, test_size=test_size, random_state=random_state)
    return trainset, testset

def build_model(trainset, k=5, user_based=False):
    # Create and fit the collaborative filtering model
    algo = KNNWithMeans(k=k, sim_options={'name': 'pearson_baseline', 'user_based': user_based})
    algo.fit(trainset)
    return algo

def make_predictions(algo, testset):
    # Make predictions on the test set
    test_pred = algo.test(testset)
    return test_pred

def evaluate_predictions(test_pred):
    # Calculate and print RMSE
    rmse = accuracy.rmse(test_pred, verbose=True)
    return rmse

# Usage example
surprise_data = load_data(data)
trainset, testset = split_data(surprise_data)
algo = build_model(trainset, k=5, user_based=False)
test_pred = make_predictions(algo, testset)
rmse = evaluate_predictions(test_pred)
print("Item-based Model: Test Set RMSE:", rmse)

##################################################
# Model-based collaborative filtering system
##################################################
"""
Bu yöntemler, makine öğrenimi ve veri madenciliği tekniklerine dayanmaktadır. 
Amaç, tahminlerde bulunabilecek modelleri eğitmektir. 
Örneğin, bir kullanıcının en çok beğenebileceği ilk 5 öğeyi tahmin edecek bir model 
eğitmek için mevcut kullanıcı-öğe etkileşimlerini kullanabiliriz. 
Bu yöntemlerin bir avantajı, bellek tabanlı yaklaşım gibi diğer yöntemlere kıyasla 
daha fazla sayıda kullanıcıya daha fazla sayıda öğe önerebilmeleridir. 
Büyük seyrek matrislerle çalışırken bile geniş kapsama alanına sahiptirler.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def create_ratings_matrix(data):
    ratings_matrix = data.pivot_table(values='rating', index='userId', columns='productId', fill_value=0)
    return ratings_matrix

def perform_svd(ratings_matrix, n_components):
    SVD = TruncatedSVD(n_components=n_components)
    decomposed_matrix = SVD.fit_transform(ratings_matrix.T)
    return decomposed_matrix

def calculate_correlation_matrix(decomposed_matrix):
    correlation_matrix = np.corrcoef(decomposed_matrix)
    return correlation_matrix

def find_similar_products(correlation_matrix, product_id, threshold, num_recommendations):
    product_names = list(correlation_matrix.index)
    product_index = product_names.index(product_id)
    
    correlation_product_ID = correlation_matrix[product_index]
    highly_correlated_indices = np.where(correlation_product_ID > threshold)[0]
    
    recommendations = list(product_names[highly_correlated_indices][:num_recommendations])
    
    return recommendations

def recommend_similar_products(data, product_id, threshold=0.75, num_recommendations=20, sample_size=20000, n_components=10):
    # Sample data
    data_sample = data.sample(sample_size)
    
    # Create ratings matrix
    ratings_matrix = create_ratings_matrix(data_sample)
    
    # Perform matrix decomposition using Truncated SVD
    decomposed_matrix = perform_svd(ratings_matrix, n_components)
    
    # Calculate correlation matrix
    correlation_matrix = calculate_correlation_matrix(decomposed_matrix)
    
    # Find highly correlated products
    recommendations = find_similar_products(correlation_matrix, product_id, threshold, num_recommendations)
    
    return recommendations

import json
import warnings
import os
import sys
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Unsupervised_Learning")))
from Hybrid_Unsupervised_Learning import Hybrid_DBSCAN
from dataframe_plot import *

current_path = os.path.dirname(os.path.abspath(__file__))
print(f"Current path: {current_path}")
# Load the dataset
dataframe = pd.read_csv(os.path.join(current_path, 'Customer_Data.csv'))
dataframe.head(7)

df_proc = dataframe.copy()
df_proc.drop(columns=['CUST_ID'], inplace=True)
df_proc.info()

# fill all null data with 0.0
df_proc.fillna(0.0, inplace=True)
df_proc.drop(columns=['PURCHASES_FREQUENCY'], inplace=True)
df_proc.drop(columns=['PURCHASES'], inplace=True)

#pca_components = len(df_proc.columns)
pca_components = 2
# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_proc)
# Perform PCA
pca = PCA(n_components=pca_components)  # full components
pca.fit(df_scaled)
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
data_preprocessed = pca.transform(df_scaled)
data_preprocessed_pd = pd.DataFrame(data_preprocessed, columns=[f'PC{i+1}' for i in range(data_preprocessed.shape[1])])

hybrid_model_params = {'DBSCAN': {'eps_quantile': 0.3, 
                                  'min_samples_quantile': 0.05, 
                                  'max_hypparams_adaptions': 7,
                                  'adapt_factor': 0.2
                                  },
                       'MeanShift': {'bw_quantile': 0.2,
                                      'max_hypparams_adaptions': 7,
                                      'adapt_factor': 0.2
                                      },
                       'Agglomerative': {'n_clusters': 4,
                                         'linkage': 'ward',
                                        },
                       'KMeans': {'n_clusters': 3
                                  },
                      }
hdbscan = Hybrid_DBSCAN(method_params=hybrid_model_params,
                        max_level=2, min_cluster_size=20, max_samples_for_bw=3000,
                        init_cluster_method='DBSCAN', sub_cluster_method='Agglomerative', noise_cluster_method='KMeans',
                        verbose=2)
data_preprocessed_pd_copy = data_preprocessed_pd.copy()
labels = hdbscan.fit_predict(data_preprocessed_pd_copy)
df_proc['HybDBSCAN_clusters'] = labels

barplot_clusters_statistics(df_proc, 'HybDBSCAN', statistic='mean', scale='linear')
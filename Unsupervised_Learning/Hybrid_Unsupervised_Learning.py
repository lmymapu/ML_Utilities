import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.cluster import DBSCAN
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from dataframe_plot import *

class Hybrid_DBSCAN(DBSCAN):
    def __init__(self, method_params=None, dataframe=None, 
                 max_level=None, min_cluster_size=None, max_samples_for_bw=None,
                 init_cluster_method='DBSCAN', sub_cluster_method='DBSCAN', noise_cluster_method='DBSCAN',
                 verbose=0, cluster_2Dplotter=None, **kwargs):
        super().__init__(**kwargs)
        self.method_params = method_params if method_params is not None else {}
        if init_cluster_method not in self.method_params.keys():
            raise ValueError(f"Initial clustering method '{init_cluster_method}' not found in method_params.")
        if sub_cluster_method not in self.method_params.keys():
            raise ValueError(f"Sub-clustering method '{sub_cluster_method}' not found in method_params.")
        if noise_cluster_method not in self.method_params.keys():
            raise ValueError(f"Noise clustering method '{noise_cluster_method}' not found in method_params.")    
        self.max_level = max_level if max_level is not None else 1
        self.min_cluster_size = min_cluster_size if min_cluster_size is not None else 1
        self.max_samples_for_bw = max_samples_for_bw if max_samples_for_bw is not None else 2
        self.initial_level = 0
        self.initial_cluster = -1
        self.init_cluster_method = init_cluster_method
        self.sub_cluster_method = sub_cluster_method
        self.noise_cluster_method = noise_cluster_method
        self.model_fit_map = {
            'DBSCAN': self.perform_DBSCAN_fit,
            'MeanShift': self.perform_MeanShift_fit,
            'Agglomerative': self.perform_agglomerative_fit,
            'KMeans': self.perform_kmeans_fit
        }
        self.verbose = verbose
        self.dataframe = dataframe
        # store some debugging/verbosing information from dataframe 
        if self.verbose >= 2 and self.dataframe is not None:
            self.comp1_range = np.min(dataframe.iloc[:, 0]), np.max(dataframe.iloc[:, 0])
            self.comp2_range = np.min(dataframe.iloc[:, 1]), np.max(dataframe.iloc[:, 1])
            self.cluster_2Dplotter = cluster_2Dplotter if cluster_2Dplotter is not None \
                                     else ClustersPlotter(figsize=(10, 6), xlim=self.comp1_range, ylim=self.comp2_range, colorstyle='Spectral')
        else:
            self.comp1_range = None
            self.comp2_range = None
            self.cluster_2Dplotter = None

    def fit_predict(self, X, y = None, **kwargs):
        self.dataframe = X.copy()
        if self.verbose >= 2:
            self.dataframe['HybDBSCAN_clusters'] = '-1'
            margin = (np.max(X.iloc[:, 0]) - np.min(X.iloc[:, 0])) * 0.05
            self.comp1_range = np.min(X.iloc[:, 0])-margin, np.max(X.iloc[:, 0])+margin
            margin = (np.max(X.iloc[:, 1]) - np.min(X.iloc[:, 1])) * 0.05
            self.comp2_range = np.min(X.iloc[:, 1])-margin, np.max(X.iloc[:, 1])+margin
            if self.cluster_2Dplotter is None:
                self.cluster_2Dplotter = ClustersPlotter(figsize=(10, 6), xlim=self.comp1_range, ylim=self.comp2_range, colorstyle='Spectral')
        # Initialize the dataframe with cluster and level columns
        self.initial_level = 0
        self.initial_cluster = -1
        df = self.initial_fit_predict(X, **kwargs)
        unique_clusters = set(df['HybDBSCAN_clusters'])

        for cluster in unique_clusters:
            cluster_last = cluster.split('_')[-1]
            if cluster_last != '-1':
                # perform further clustering for subclusters
                cluster_indices = df[df['HybDBSCAN_clusters'] == cluster].index
                df.loc[cluster_indices] = self.subcluster_fit_predict(X.loc[cluster_indices], self.initial_level+1, cluster, **kwargs)
            else:
                # perform further clustering for noise clusters
                cluster_indices = df[df['HybDBSCAN_clusters'] == cluster].index
                df.loc[cluster_indices] = self.noise_fit_predict(X.loc[cluster_indices], self.initial_level+1, cluster, **kwargs)

        labels = df['HybDBSCAN_clusters'].values
        self.dataframe['HybDBSCAN_levels'] = df['HybDBSCAN_levels']
        self.dataframe['HybDBSCAN_clusters'] = df['HybDBSCAN_clusters']
        if self.verbose >= 2:
            print("Clustering Completed.")
            self.cluster_2Dplotter.plot2D_clusters(self.dataframe, self.dataframe.columns[0], self.dataframe.columns[1], 'HybDBSCAN_clusters', title="Final Clustering")
        return labels
    
    def initial_fit_predict(self, X, **kwargs):
        df = X.copy()
        if self.verbose >= 1:
            print(f"level {self.initial_level} Initial Clustering - Samples counts: {df.shape[0]}")
        df = self.model_fit_map[self.init_cluster_method](df, self.initial_level, '', **kwargs)
        return df
 
    
    def subcluster_fit_predict(self, X, init_level, init_cluster, **kwargs):
        df = X.copy()
        if init_level >= self.max_level:
            print(f"    Max level {self.max_level} for cluster {init_cluster} reached. Leaf-Cluster found, size = {df.shape[0]}.")
            df = self.perform_leaf_cluster_padding(df, init_level, init_cluster)
        elif df.shape[0] <= self.min_cluster_size: 
            print(f"    Cluster {init_cluster} has too few samples ({df.shape[0]}). Leaf-Cluster found, size = {df.shape[0]}.")
            df = self.perform_leaf_cluster_padding(df, init_level, init_cluster)
        else:
            if self.verbose >= 1:
                print(f"level {init_level} Sub-Cluster {init_cluster} - Samples counts: {df.shape[0]}")
            # perform clustering for the current cluster
            df = self.model_fit_map[self.sub_cluster_method](df, init_level, init_cluster, **kwargs)
            unique_clusters = set(df['HybDBSCAN_clusters'])
            # perform further clustering iteratively for each sub
            for cluster in unique_clusters:
                cluster_indices = df[df['HybDBSCAN_clusters'] == cluster].index
                df.loc[cluster_indices] = self.subcluster_fit_predict(X.loc[cluster_indices], init_level + 1, cluster, **kwargs)
        
        return df

    def noise_fit_predict(self, X, init_level, init_cluster, **kwargs):
        df = X.copy()
        if init_level >= self.max_level:
            print(f"    Max level {self.max_level} for noise cluster {init_cluster} reached. Leaf-Cluster found, size = {df.shape[0]}.")
            df = self.perform_leaf_cluster_padding(df, init_level, init_cluster)
        elif df.shape[0] <= self.min_cluster_size:
            print(f"    Noise cluster {init_cluster} has too few samples ({df.shape[0]}). Leaf-Cluster found, size = {df.shape[0]}.")
            df = self.perform_leaf_cluster_padding(df, init_level, init_cluster)
        else:                
            if self.verbose >= 1:
                print(f"level {init_level} Noise Cluster {init_cluster} - Samples counts: {df.shape[0]}")
            # perform DBSCAN for noise
            #df = self.perform_DBSCAN_fit(df, init_level, init_cluster, **kwargs)
            df = self.model_fit_map[self.noise_cluster_method](df, init_level, init_cluster, **kwargs)
            unique_clusters = set(df['HybDBSCAN_clusters'])

            # perform further clustering iteratively
            for cluster in unique_clusters:
                cluster_last = cluster.split('_')[-1]
                if cluster_last != '-1':
                    # for non-noise clusters, perform sub-clustering
                    cluster_indices = df[df['HybDBSCAN_clusters'] == cluster].index
                    df.loc[cluster_indices] = self.subcluster_fit_predict(X.loc[cluster_indices], init_level + 1, cluster, **kwargs)
                else:
                    # for noise clusters, perform noise clustering
                    cluster_indices = df[df['HybDBSCAN_clusters'] == cluster].index
                    df.loc[cluster_indices] = self.noise_fit_predict(X.loc[cluster_indices], init_level + 1, cluster, **kwargs)

        return df
    
    def perform_DBSCAN_fit(self, df, init_level, init_cluster, **kwargs):
        # estimate the epsilon
        bw_min = 1e-5
        samples_estimated = np.min([df.shape[0], self.max_samples_for_bw])
        eps_quantile = self.method_params['DBSCAN'].get('eps_quantile') \
                        if 'eps_quantile' in self.method_params['DBSCAN'].keys() \
                        else 0.3
        bw = estimate_bandwidth(df, quantile=eps_quantile, n_samples=samples_estimated)
        # estimate the core size
        min_samples_quantile = self.method_params['DBSCAN'].get('min_samples_quantile') \
                                if 'min_samples_quantile' in self.method_params['DBSCAN'].keys() \
                                else 0.3
        n_clu = np.int32(df.shape[0] * min_samples_quantile) + 1

        if bw < bw_min:
            if self.verbose >= 1:
                print(f"    Estimated bandwidth {bw:.6f} is too small. Leaf-Cluster found, size = {df.shape[0]}.")
            df['HybDBSCAN_levels'] = init_level
            df['HybDBSCAN_clusters'] = f"{init_cluster}"          
        elif n_clu <= 1:
            if self.verbose >= 1:
                print(f"    Estimated min samples equals 1, no more clustering possible. Leaf-Cluster found, size = {df.shape[0]}.")
            df['HybDBSCAN_levels'] = init_level
            df['HybDBSCAN_clusters'] = f"{init_cluster}"
        else:
            if self.verbose >= 1:            
                print(f"    Start new DBSCAN clustering, Estimated epsilon: {bw:.3f}, core samples size: {n_clu}")
            # Perform DBSCAN clustering
            labels_complete = []
            adapt_cnt = 0
            max_adapts = self.method_params['DBSCAN'].get('max_hypparams_adaptions') \
                        if 'max_hypparams_adaptions' in self.method_params['DBSCAN'].keys() \
                        else 1
            adapt_fac = self.method_params['DBSCAN'].get('adapt_factor') \
                        if 'adapt_factor' in self.method_params['DBSCAN'].keys() \
                        else 0.2

            while (len(set(labels_complete)) < 2):
                labels_complete = []
                dbscan = DBSCAN(eps=bw, min_samples=n_clu, n_jobs=-1, **kwargs)
                labels = dbscan.fit_predict(df)
                
                for l in labels:
                    new_label = f"{init_cluster}_{l}"
                    labels_complete.append(new_label)

                # check if labels contains only -1
                if len(set(labels)) == 1 and (-1 in set(labels)):
                    # if all points are noise, decrease core size
                    adapt_cnt += 1
                    bw *= (1 + adapt_fac)
                    n_clu = np.int32(n_clu / (1 + adapt_fac)) + 1
                    adapt_fac *= adapt_fac         # decrease adapt factor for next iteration
                    if self.verbose >= 1:
                        print(f"    All points are noise, increase epsilon to {bw:.3f} and decrease min samples to {n_clu}.")
                elif len(set(labels)) == 1 and (0 in set(labels)):
                    # if all points are in one cluster, increase core size
                    adapt_cnt += 1
                    bw = bw / (1 + adapt_fac)
                    n_clu = np.int32(n_clu * (1 + adapt_fac))
                    adapt_fac *= adapt_fac         # decrease adapt factor for next iteration
                    if self.verbose >= 1:
                        print(f"    All points are in one cluster, decrease epsilon to {bw:.3f} and increase min samples to {n_clu}.")   
                else:
                    # if we have a valid clustering, stop the loop
                    if self.verbose >= 1:
                        print(f"    Found {len(set(labels))} clusters.")

                if adapt_cnt >= max_adapts:
                    if self.verbose >= 1:
                        print(f"    Max adaptions {max_adapts} reached.")
                    adapt_cnt = 0       
                    break
            df['HybDBSCAN_levels'] = init_level + 1
            df['HybDBSCAN_clusters'] = labels_complete
            if self.verbose >= 2:
                index_label_updated = df.index
                self.dataframe.loc[index_label_updated, 'HybDBSCAN_clusters'] = labels_complete
                self.cluster_2Dplotter.plot2D_clusters(self.dataframe, self.dataframe.columns[0], self.dataframe.columns[1], 'HybDBSCAN_clusters', 
                                                       title=f"Level {init_level} - Splitting from cluster {init_cluster} - DBSCAN")
        return df
    
    def perform_MeanShift_fit(self, df, init_level, init_cluster, **kwargs):
        # estimate the bandwidth
        bw_min = 1e-5
        samples_estimated = np.min([df.shape[0], self.max_samples_for_bw])
        bw_quantile = self.method_params['MeanShift'].get('bw_quantile') \
                        if 'bw_quantile' in self.method_params['MeanShift'].keys() \
                        else 0.3
        bw = estimate_bandwidth(df, quantile=bw_quantile, n_samples=samples_estimated)
        if bw < bw_min:
            if self.verbose >= 1:
                print(f"    Estimated bandwidth {bw:.6f} is too small. Leaf-Cluster found.")
            df['HybDBSCAN_levels'] = init_level
            df['HybDBSCAN_clusters'] = f"{init_cluster}"
        else:        
            # perform MeanShift clustering
            if self.verbose >= 1:
                print(f"    Start new MeanShift clustering, Estimated bandwidth: {bw:.3f}")
            labels_complete = []
            adapt_cnt = 0
            max_adapts = self.method_params['MeanShift'].get('max_hypparams_adaptions') \
                        if 'max_hypparams_adaptions' in self.method_params['MeanShift'].keys() \
                        else 1
            adapt_fac = self.method_params['MeanShift'].get('adapt_factor') \
                        if 'adapt_factor' in self.method_params['MeanShift'].keys() \
                        else 0.2

            while (len(set(labels_complete)) < 2):
                labels_complete = []
                meanshift = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1, **kwargs)
                labels = meanshift.fit_predict(df)
                for l in labels:
                    new_label = f"{init_cluster}_{l}"
                    labels_complete.append(new_label)
                
                if len(set(labels)) == 1:
                    # if only 1 cluster is found, decrease bandwidth
                    adapt_cnt += 1
                    bw = bw / (1 + adapt_fac)
                    if self.verbose >= 1:
                        print(f"    All points are in one cluster, decrease bandwidth to {bw:.6f}.")
                else:
                    # if we have a valid clustering, stop the loop
                    if self.verbose >= 1:
                        print(f"    Found {len(set(labels))} clusters.")
                
                if adapt_cnt >= max_adapts:
                    if self.verbose >= 1:
                        print(f"    Max adaptions {max_adapts} reached.")
                    break

            df['HybDBSCAN_levels'] = init_level + 1
            df['HybDBSCAN_clusters'] = labels_complete
            if self.verbose >= 2:
                index_label_updated = df.index
                self.dataframe.loc[index_label_updated, 'HybDBSCAN_clusters'] = labels_complete
                self.cluster_2Dplotter.plot2D_clusters(self.dataframe, self.dataframe.columns[0], self.dataframe.columns[1], 'HybDBSCAN_clusters', 
                                                       title=f"Level {init_level} - Splitting from cluster {init_cluster} - MeanShift")
        return df

    def perform_agglomerative_fit(self, df, init_level, init_cluster, **kwargs):
        if self.verbose >= 1:
            print(f"    Start new Hierarchical Agglomerative clustering")

        # perform agglomerative clustering
        ag = AgglomerativeClustering(**self.method_params['Agglomerative'])
        labels = ag.fit_predict(df)
        labels_complete = []
        for l in labels:
            new_label = f"{init_cluster}_{l}"
            labels_complete.append(new_label)
        df['HybDBSCAN_levels'] = init_level + 1
        df['HybDBSCAN_clusters'] = labels_complete
        if self.verbose >= 2:
            index_label_updated = df.index
            self.dataframe.loc[index_label_updated, 'HybDBSCAN_clusters'] = labels_complete
            self.cluster_2Dplotter.plot2D_clusters(self.dataframe, self.dataframe.columns[0], self.dataframe.columns[1], 'HybDBSCAN_clusters', 
                                                   title=f"Level {init_level} - Splitting from cluster {init_cluster} - Agglomerative")
        return df
    
    def perform_kmeans_fit(self, df, init_level, init_cluster, **kwargs):
        if self.verbose >= 1:
            print(f"    Start new KMeans clustering")

        # perform k-means clustering
        kmeans = KMeans(**self.method_params['KMeans'])
        labels = kmeans.fit_predict(df)
        labels_complete = []
        for l in labels:
            new_label = f"{init_cluster}_{l}"
            labels_complete.append(new_label)
        df['HybDBSCAN_levels'] = init_level + 1
        df['HybDBSCAN_clusters'] = labels_complete
        if self.verbose >= 2:
            index_label_updated = df.index
            self.dataframe.loc[index_label_updated, 'HybDBSCAN_clusters'] = labels_complete
            self.cluster_2Dplotter.plot2D_clusters(self.dataframe, self.dataframe.columns[0], self.dataframe.columns[1], 'HybDBSCAN_clusters', 
                                                   title=f"Level {init_level} - Splitting from cluster {init_cluster} - KMeans")
        return df

    def perform_leaf_cluster_padding(self, df, init_level, init_cluster):
        # perform padding for leaf clusters
        df['HybDBSCAN_levels'] = init_level
        df['HybDBSCAN_clusters'] = f"{init_cluster}"
        return df
    

    


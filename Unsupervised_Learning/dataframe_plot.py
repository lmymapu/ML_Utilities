import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ClustersPlotter:
    def __init__(self, figsize=None, xlim=None, ylim=None, colorstyle='viridis'):
        """
        Initialize the ClustersPlotter with the given parameters.
        
        Parameters:
        figsize (tuple): Size of the figure. Default is None.
        xlim (tuple): Limits for x-axis. Default is None.
        ylim (tuple): Limits for y-axis. Default is None.
        colorstyle (str): Color style for the plot. Default is 'viridis'.
        title (str): Title of the plot. Default is None.
        """
        self.figsize = figsize if figsize else (10, 6)
        self.xlim = xlim if xlim else (-16, 20)
        self.ylim = ylim if ylim else (-16, 20)
        self.colorstyle = colorstyle
        self.cluster_color_assignments = None
        self.unique_clusters = None

    def plot2D_clusters(self, dataframe, comp1, comp2, cluster_col_name, title=None):
        """
        Plot 2D clusters of the data points.
        
        Parameters:
        dataframe (pd.DataFrame): DataFrame containing the data points and cluster labels.
        comp1 (str): Name of the first component for x-axis.
        comp2 (str): Name of the second component for y-axis.
        algorithm (str): Name of the clustering algorithm used.
        """
        plt.figure(figsize=self.figsize)
        unique_clusters = set(dataframe[cluster_col_name])
        # Create an evenly distributed RGB colormap for all unique clusters
        self.update_colormap(unique_clusters)
        #colors = {cluster: plt.cm.tab10(i) for i, cluster in enumerate(unique_clusters)}
        plt.scatter(dataframe[comp1], dataframe[comp2], c=dataframe[cluster_col_name].map(self.cluster_color_assignments), cmap='viridis', alpha=0.6)
        # turn on legend
        if title is None:
            plt.title(f"Scatter Plot for Components {comp1} and {comp2} with Clusters Labeled")
        else:
            plt.title(title)
        plt.xlabel(comp1)
        plt.ylabel(comp2)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)

        # create a legend for the clusters
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.cluster_color_assignments[cluster], markersize=10, label=f' {cluster}') for cluster in unique_clusters]
        plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(0.96, 1), loc='upper left')
        plt.show()

    def update_colormap(self, clusters_set):
        if self.unique_clusters is None:
            self.unique_clusters = clusters_set
            num_clusters = len(self.unique_clusters)
            colormap = plt.cm.get_cmap(self.colorstyle, num_clusters)
            self.cluster_color_assignments = {cluster: colormap(i) for i, cluster in enumerate(self.unique_clusters)}
        else:
            # Check if the new clusters are already in the existing colormap
            old_cluster_color_assignments = self.cluster_color_assignments.copy()
            new_clusters = clusters_set - self.unique_clusters
            lost_clusters = self.unique_clusters - clusters_set
            common_clusters = self.unique_clusters & clusters_set
            # delete lost clusters from the colormap
            for cluster in lost_clusters:
                del self.cluster_color_assignments[cluster]
            # add new clusters to the colormap
            num_clusters = len(common_clusters) + len(new_clusters) + len(lost_clusters)
            colormap = plt.cm.get_cmap(self.colorstyle, num_clusters)
            new_colors_list = []

            for i, cluster in enumerate(common_clusters):
                self.cluster_color_assignments[cluster] = old_cluster_color_assignments[cluster]
            for i in range(0, num_clusters):
                if colormap(i) not in old_cluster_color_assignments.values():
                    new_colors_list.append(colormap(i))
            # assign new colors to new clusters
            for i, cluster in enumerate(new_clusters):
                self.cluster_color_assignments[cluster] = new_colors_list[i]
            
            # update the unique clusters set
            self.unique_clusters = clusters_set

def barplot_clusters_statistics(dataframe, algorithm, statistic='mean', scale='linear', title=None):
    """
    Generate a bar plot to visualize cluster statistics.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing the data points and cluster labels.
    algorithm (str): Name of the clustering algorithm used.
    statistic (str): Statistic to display ('mean', 'std', 'count'). Default is 'mean'.
    scale (str): Scale of the y-axis ('linear', 'log', 'log1p'). Default is 'linear'.
    title (str): Title of the plot. If None, a default title is generated. Default is None.
    """
    # clean up excessive cluster columns other than the dedicated algorithm
    unique_columns = [col for col in dataframe.columns if '_clusters' not in col]
    df = dataframe[unique_columns].copy()
    df[algorithm+'_clusters'] = dataframe[algorithm+'_clusters']

    grouped_stats = df.groupby(algorithm+'_clusters').agg(['mean', 'std', 'count'])
    shown_columns = [col for col in grouped_stats.columns if statistic in col]
    num_columns = len(shown_columns)
    colormap = plt.cm.get_cmap('Spectral', num_columns)
    colors = {col: colormap(i) for i, col in enumerate(shown_columns)}
    # bar plot of the mean values of each cluster
    #plt.figure(figsize=(15, 8))
    
    if scale == 'linear':
        grouped_stats[shown_columns].plot(kind='bar', alpha=0.7, color=colors, figsize=(15, 8))
    elif scale == 'log':
        grouped_stats[shown_columns].applymap(lambda x: np.log(x)).plot(kind='bar', alpha=0.7, color=colors, figsize=(15, 8))
    elif scale == 'log1p':
        grouped_stats[shown_columns].applymap(lambda x: np.log1p(x)).plot(kind='bar', alpha=0.7, color=colors, figsize=(15, 8))
    else:
        raise ValueError("Invalid scale. Choose 'linear', 'log', or 'log1p'.")
    if title is None:
        plt.title(f"{scale.capitalize()}-Scaled {statistic.capitalize()} Values of Each Cluster by {algorithm.capitalize()}")
    else:
        plt.title(title)
    plt.xlabel('Clusters')
    plt.ylabel(f'{statistic.capitalize()} Values')
    plt.legend([col[0] for col in shown_columns], loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.show()
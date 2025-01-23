# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
#
# # Define your data
# data_dict = {
#     'domain_0': np.random.rand(10, 10),
#     'domain_1':  np.random.rand(10, 10),
#     'domain_2':  np.random.rand(10, 10),
#     'domain_3':  np.random.rand(10, 10),
# }
#
# label_dict = {
#     'domain_0': np.random.randint(low=0, high=2, size=10),
#     'domain_1': np.random.randint(low=0, high=2, size=10),
#     'domain_2': np.random.randint(low=0, high=2, size=10),
#     'domain_3': np.random.randint(low=0, high=2, size=10)
# }
#
# # Define the icons for each class
# icons = ['o', 's']
#
# # Define the colors for each domain
# colors = ['red', 'blue', 'green', 'orange']
#
# # Define the t-SNE parameters
# perplexity = 30
# learning_rate = 200
#
# # Initialize the t-SNE model
# tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, init='pca')
#
# # Plot the t-SNE embeddings with different icons and colors for each domain and class
# plt.figure(figsize=(8, 8))
# for domain_idx, (domain_name, data) in enumerate(data_dict.items()):
#     labels = label_dict[domain_name]
#     for class_idx, icon in enumerate(icons):
#         class_mask = labels == class_idx
#         plot_mask = class_mask
#         plt.scatter(tsne.fit_transform(data[plot_mask])[:,0], tsne.fit_transform(data[plot_mask])[:,1], c=colors[domain_idx], marker=icon, label=f'{domain_name}, Class {class_idx}')
# plt.legend()
# plt.show()
# import IPython;IPython.embed()


import numpy as np
import matplotlib.pyplot as plt
import umap

# Define your data and labels
data_dict = {
    'domain_0': np.random.rand(10, 10),
    'domain_1':  np.random.rand(10, 10),
    'domain_2':  np.random.rand(10, 10),
    'domain_3':  np.random.rand(10, 10),
}

label_dict = {
    'domain_0': np.random.randint(low=0, high=2, size=10),
    'domain_1': np.random.randint(low=0, high=2, size=10),
    'domain_2': np.random.randint(low=0, high=2, size=10),
    'domain_3': np.random.randint(low=0, high=2, size=10)
}

# Define the icons for each class
icons = ['o', 's']

# Define the colors for each domain
colors = ['red', 'blue', 'green', 'orange']

# Initialize the UMAP model
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1)

# Plot the UMAP embeddings with different icons and colors for each domain and class
plt.figure(figsize=(8, 8))
for domain_idx, (domain_name, data) in enumerate(data_dict.items()):
    labels = label_dict[domain_name]
    for class_idx, icon in enumerate(icons):
        class_mask = labels[:, 1] == class_idx
        plot_mask = class_mask
        plt.scatter(umap_model.fit_transform(data[plot_mask]), c=colors[domain_idx], marker=icon, label=f'{domain_name}, Class {class_idx}')
plt.legend()
plt.show()
import IPython;IPython.embed()
"""Unsupervised clustering on learned encoding with NCE
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import ast

from sklearn.cluster import KMeans
from tqdm import tqdm
from kneed import KneeLocator
from PIL import Image


parser = argparse.ArgumentParser(description="Clustering on NCE representations")
parser.add_argument("--n-features", type=int, default=128, help="Representation size.")
parser.add_argument('-f', '--csv-file', default='./data/featuresntags.csv',
                    help='Path to csv containing representations')
parser.add_argument("--nb-clusters", default=13, type=int, help="Number of clusters.")
parser.add_argument("--elbow", action="store_true", help="Use elbow method to find nb_clusters.")
parser.add_argument("--nb-max-clusters", type=int, default=50,  
                    help="Maximum number of clusters to consider for elbow method.")
parser.add_argument("--kmeans", action="store_true", default="Run and save kmeans on data.")
parser.add_argument("--show-proto", action="store_true", help="Display images close de kmeans centers.")
parser.add_argument("--nb-proto", type=int, default=5, help="Number of prototypes to display per class.")
parser.add_argument("--show-tags", action="store_true", help="Print most popular tags in clusters.")
parser.add_argument("--nb-tags", type=int, help="Number of tags to show per cluster.")
parser.add_argument("--load", default="./checkpoint/kmeans_7.pkl", help="Path/to/the/model/to/load")
parser.add_argument("--cluster-hist", action="store_true", help="Save cluster hist as png.")


args = parser.parse_args()

df = pd.read_csv(args.csv_file)

features_mat = df.loc[:,[str(i) for i in range(  args.n_features)]].to_numpy()

km_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 1
}

def main(): 
    if args.elbow:
        # elbow method
        sse = []
        for k in tqdm(range(1, args.nb_max_clusters)):
            km = KMeans(n_clusters=k, **km_kwargs)
            km.fit(features_mat)
            sse.append(km.inertia_)

        kneedle = KneeLocator(range(1, args.nb_max_clusters), sse, S=1.0, curve="convex", direction="decreasing")

        print(f"The appropriated number of clusters value should be : {round(kneedle.elbow, 3)}")

        plt.style.use("fivethirtyeight")
        plt.plot(range(1, args.nb_max_clusters), sse)
        plt.xticks(range(1, args.nb_max_clusters))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.savefig(f"elbow_curve_{args.nb_max_clusters}.png")

    if args.kmeans:
        km = KMeans(n_clusters=args.nb_clusters, **km_kwargs)   
        km.fit(features_mat)

        pkl_filename = f"kmeans_{args.nb_clusters}.pkl"

        with open(pkl_filename, 'wb') as file:
            pickle.dump(km, file)

    if args.show_proto:
        if len(args.load) > 1:
            model = load_model(args.load)
        else:
            raise Warning("Please specify a model to load.")
        show_centers_prototypes(model, args.nb_proto)

    if args.show_tags:
        if len(args.load) > 1:
            model = load_model(args.load)
        else:
            raise Warning("Please specify a model to load.")
        show_tags(model, args.nb_tags)

    if args.cluster_hist:
        if len(args.load) > 1:
            model = load_model(args.load)
        else:
            raise Warning("Please specify a model to load.")
        show_cluster_hist(model)


def show_tags(model, nb_tags):
    df['cluster'] = pd.Series(model.predict(features_mat))
    dfgp = df.groupby(['cluster'])
    tags_concat = []
    for tags in df['tags']:
        if tags != "No tags":
            tags_concat += ast.literal_eval(tags)
    df_tags_all = pd.DataFrame(tags_concat)
    counts_all = df_tags_all.value_counts()
    
    # print(counts_all)
    counts_all[counts_all > 250].plot(kind='bar')
    plt.savefig("counts_tags.png")
    for cluster_id, group in dfgp:
        tags_concat = []
        for tags in group['tags']:
            if tags != "No tags":
                tags_concat += ast.literal_eval(tags)
        df_tags = pd.DataFrame(tags_concat, columns=[f"tag for cluster {cluster_id}"])
        counts_cluster = df_tags.value_counts().astype(float)
        for key, value in counts_cluster.items():
            if counts_all[key] < 250:
                counts_cluster[key] = 0
            else:
                counts_cluster[key] = value / counts_all[key]
        print(f"CLUSTER {cluster_id}")
        print(counts_cluster.sort_values(ascending=False).head(10))    


def show_cluster_hist(model):
    centers = model.cluster_centers_
    for i, center in enumerate(centers):
        similarity_vector = pd.Series(features_mat.dot(center))
        plt.subplot(5, 3, i+1)
        similarity_vector.hist(bins=30)
    plt.savefig(f"cluster_hists.png")


def show_centers_prototypes(model, nb_proto):
    centers = model.cluster_centers_
    for i, center in enumerate(centers):
        im_idexes = get_closest(center, nb_proto)
        for j, im_idx in enumerate(im_idexes):
            plot_idx = i*nb_proto + j + 1
            print(plot_idx)
            ax = plt.subplot(len(centers), nb_proto, plot_idx)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if not j:
                ax.set_ylabel(f"C{i}")
            plt.imshow(Image.open(df['img_path'][im_idx]), aspect="auto")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(f'cluster_visu.png', bbox_inches='tight', dpi=200)
    plt.tight_layout(pad=0)


def save_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(model_path):
    with open(model_path, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model


def get_closest(query_point, nb_im):
    similarity_vector = pd.Series(features_mat.dot(query_point))
    similarity_vector.sort_values(inplace = True, ascending=False)
    idx_list = []
    for i in range(nb_im):
        # plt.subplot(1, nb_neighbours, i)
        idx = similarity_vector.index[i]
        # sim_value = similarity_vector[idx]
        # print(sim_value)
        # show_by_idx(idx)
        idx_list.append(idx)
    return idx_list


if __name__ == '__main__':
    main()
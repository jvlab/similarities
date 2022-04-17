import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.util import stimulus_names
from sklearn.decomposition import PCA
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def stretch_axes(points):
    """
    Run PCA.
    Stretch axes so as to make the variance along each axis the same.
    Do so by dividing the values of the coordinate by the standard deviation of values along that axis.
    This way points along each axis will have unit variance. This does not affect the radial
    distribution of points, only their distance from the origin in different directions.
    """
    n_components = 5
    pca = PCA(n_components=n_components)
    # obtain the 5 PC directions and project data onto that space
    temp = pca.fit_transform(points)
    # normalize each axis by its standard deviation to make sd across each axis the same
    for i in range(n_components):
        temp[:, i] = temp[:, i] / np.std(temp[:, i])
    points = temp
    return points


def scatterplots_2d_annotated(subject_name, subject_exp_data, pc1=1, pc2=2):
    sns.set_style('darkgrid')
    stimuli = stimulus_names()

    fig, ax = plt.subplots()
    plt.scatter(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1], c="#31505A", marker='o')
    # add labels to points
    label_idx = 0
    for x, y in zip(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1]):
        plt.annotate(stimuli[label_idx],  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 1.5),  # distance from text to points (x,y)
                     size=10,
                     ha='center')  # horizontal alignment can be left, right or center
        label_idx += 1
    plt.xlabel('Principal Component {}'.format(pc1))
    plt.ylabel('Principal Component {}'.format(pc2))
    plt.title(subject_name)
    plt.axis('square')
    ax.set_xlim(-2, 3.5)
    ax.set_ylim(-2, 3.5)
    plt.show()


def scatterplots_2d_colored(subject_name, subject_exp_data, pc1=1, pc2=2, colorby='land'):
    sns.set_style('darkgrid')
    stimuli = stimulus_names()
    maps = {}
    # define color scheme here for now
    categories = [0, 0, 0, 0] + [1]*28 + [0, 1, 1, 1, 0]
    # colormap by category - color water dwelling and land (non-water) dwelling
    colormap = np.array(['blue', 'brown'])
    maps['land'] = colormap[categories]
    categories2 = [0, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5,
                   5, 5, 4, 2, 4, 4, 4]
    # color mammals 0, fish 1, amphibians 2, birds 3, reptiles 4, insects + snail 5 separately
    colormap2 = np.array(['brown', 'blue', 'green', 'yellow', 'purple', 'black'])
    maps['kingdom'] = colormap2[categories2]
    sizes = np.log(np.array([7.4, 0.17, 15, 100, 0.6, 2, 2.5, 2, 1.5, 0.5, 2.7, 0.3, 0.6, 0.25, 4.5, 1.5, 2, 1.5, 2, 3.5, 3, 5.5, 5,
             18, 10, 2, 9, 0.017, 0.2, 0.03, 0.03, 0.08, 3.5, 0.3, 0.5, 3.5, 10]))  # color coarsely by size (in ft) Google
    sizes = sizes/max(sizes)
    rainbow = cm.get_cmap('gnuplot')
    maps['size'] = [rainbow(val) for val in sizes]
    animal_colors = ['#778899', '#ffa500', '#6699cc', '#6699cc', '#6495ED', '#4a4300', '#341c02', '#7f7053', '#2f4f4f',
                     '#847D6F', '#5D4333', 'white', 'gray', '#5D4333', '#835C3B', 'gray', '#C4A484', '#C35817', '#eae0c8',
                     '#efdfbb', '#202020', '#a13d2d', '#8A2D1C', '#956201', '#9e9b90', 'brown', '#D49C4A', 'black', 'orange',
                     'red', 'black', '#5D4333', '#BC815F', '#a69d86', '#c4be6c', '#254117', '#73A16C']  # color by actual color of animal (by eye)
    maps['color'] = animal_colors

    colors = maps[colorby]
    fig, ax = plt.subplots()
    plt.scatter(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1], c=colors, marker='o')
    # add labels to points
    label_idx = 0
    for x, y in zip(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1]):
        plt.annotate(stimuli[label_idx],  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 1.5),  # distance from text to points (x,y)
                     size=8,
                     ha='center')  # horizontal alignment can be left, right or center
        label_idx += 1
    plt.xlabel('Principal Component {}'.format(pc1))
    plt.ylabel('Principal Component {}'.format(pc2))
    plt.title(subject_name)
    plt.axis('square')
    ax.set_xlim(-2, 3.5)
    ax.set_ylim(-2, 3.5)
    plt.show()


if __name__ == '__main__':
    PATH_TO_NPY_FILE = input("Path to npy file containing 5D coordinates "
                             "(e.g., ./sample-materials/subject-data/geometry-fitting/S7/"
                             "S7_word_anchored_points_sigma_0.18_dim_5.npy): ")
    NAME = input("Subject name or ID (e.g., S7): ")
    data = np.load(PATH_TO_NPY_FILE)
    data = stretch_axes(data)
    COLORBY = ['size', 'land', 'kingdom', 'color']
    for feature in COLORBY:
        scatterplots_2d_colored(NAME, data, colorby=feature)




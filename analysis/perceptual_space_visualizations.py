import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from analysis.util import stimulus_names
from sklearn.decomposition import PCA


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
    plt.scatter(subject_exp_data[:, pc1 - 1], subject_exp_data[:, pc2 - 1], c="#31505A", marker='.')
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


if __name__ == '__main__':
    PATH_TO_NPY_FILE = input("Path to npy file containing 5D coordinates "
                             "(e.g., ./sample-materials/subject-data/geometry-fitting/S7/"
                             "S7_word_anchored_points_sigma_0.18_dim_5.npy): ")
    NAME = input("Subject name or ID (e.g., S7): ")
    data = np.load(PATH_TO_NPY_FILE)
    data = stretch_axes(data)
    scatterplots_2d_annotated(NAME, data)

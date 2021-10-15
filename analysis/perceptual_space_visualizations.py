import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import skewnorm
from mpl_toolkits.mplot3d import Axes3D

mc = {'name': 'MC'}
saw = {'name': 'SAW'}
ycl = {'name': 'YCL'}
bl = {'name': 'BL'}
efv = {'name': 'EFV'}
sj = {'name': 'SJ'}
jf = {'name': 'JF'}
nk = {'name': 'NK'}
sa = {'name': 'SA'}

PATH = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/simulations/euclidean/analysis_real_data'
# mc['texture'] = np.load(PATH + '/MC/MC_texture_points_sigma_0.18_dim_5.npy')
# mc['intermediate_texture'] = np.load(PATH + '/MC/MC_intermediate_texture_points_sigma_0.18_dim_5.npy')
# mc['intermediate_object'] = np.load(PATH + '/MC/MC_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# mc['image'] = np.load(PATH + '/MC/MC_image_anchored_points_sigma_0.18_dim_5.npy')
# mc['word'] = np.load(PATH  + '/MC/MC_word_anchored_points_sigma_0.18_dim_5.npy')
# mc['texture_grayscale'] = np.load(PATH + '/MC/MC_texture_grayscale_anchored_points_sigma_0.18_dim_5.npy')
# mc['color'] = np.load(PATH + '/MC/MC_texture_color_anchored_points_sigma_0.2_dim_5.npy')
# mc['color_old'] = np.load(PATH + '/MC/MC_texture_color_anchored_points_sigma_0.18_dim_5.npy')

# saw['word'] = np.load(PATH  + '/SAW/SAW_word_anchored_points_sigma_0.18_dim_5.npy')
# saw['texture'] = np.load(PATH  + '/SAW/SAW_texture_points_sigma_0.18_dim_5.npy')
# saw['intermediate_texture'] = np.load(PATH  + '/SAW/SAW_intermediate_texture_anchored_points_sigma_0.18_dim_5.npy')
# saw['intermediate_object'] = np.load(PATH  + '/SAW/SAW_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# saw['color'] = np.load(PATH + '/SAW/SAW_texture_color_anchored_points_sigma_0.16_dim_5.npy')
# saw['texture_grayscale'] = np.load(PATH + '/SAW/SAW_texture_grayscale_anchored_points_sigma_0.18_dim_5.npy')
saw['image'] = np.load(PATH  + '/SAW/SAW_image_anchored_points_sigma_0.18_dim_5.npy')

#
# ycl['word'] = np.load(PATH  + '/YCL/YCL_word_anchored_points_sigma_0.18_dim_5.npy')
# ycl['texture'] = np.load(PATH  + '/YCL/YCL_texture_anchored_points_sigma_0.18_dim_5.npy')
# ycl['intermediate_texture'] = np.load(PATH  + '/YCL/YCL_intermediate_texture_points_sigma_0.18_dim_5.npy')
# ycl['intermediate_object'] = np.load(PATH  + '/YCL/YCL_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# ycl['image'] = np.load(PATH + '/YCL/YCL_image_anchored_points_sigma_0.18_dim_5.npy')
#
# bl['intermediate_texture'] = np.load(PATH  + '/BL/BL_intermediate_texture_anchored_points_sigma_0.18_dim_5.npy')
# bl['intermediate_object'] = np.load(PATH  + '/BL/BL_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# bl['texture'] = np.load(PATH  + '/BL/BL_texture_anchored_points_sigma_0.18_dim_5.npy')
# bl['word'] = np.load(PATH  + '/BL/BL_word_anchored_points_sigma_0.18_dim_5.npy')
#
# efv['intermediate_texture'] = np.load(PATH  + '/EFV/EFV_intermediate_texture_anchored_points_sigma_0.18_dim_5.npy')
# efv['intermediate_object'] = np.load(PATH  + '/EFV/EFV_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# efv['word'] = np.load(PATH  + '/EFV/EFV_word_anchored_points_sigma_0.18_dim_5.npy')
# efv['image'] = np.load(PATH  + '/EFV/EFV_image_anchored_points_sigma_0.18_dim_5.npy')

# sj['intermediate_texture'] = np.load(PATH  + '/SJ/SJ_intermediate_texture_anchored_points_sigma_0.18_dim_5.npy')
# sj['intermediate_object'] = np.load(PATH  + '/SJ/SJ_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# sj['word'] = np.load(PATH  + '/SJ/SJ_word_anchored_points_sigma_0.18_dim_5.npy')

# jf['texture'] = np.load(PATH  + '/JF/JF_texture_anchored_points_sigma_0.18_dim_5.npy')
# jf['word'] = np.load(PATH + '/JF/JF_word_anchored_points_sigma_0.18_dim_5.npy')

# nk['color'] = np.load(PATH + '/NK/NK_texture_color_anchored_points_sigma_0.2_dim_5.npy')
# nk['color_old'] = np.load(PATH + '/NK/NK_texture_color_anchored_points_sigma_0.18_dim_5.npy')
# nk['intermediate_object'] = np.load(PATH  + '/NK/NK_intermediate_object_anchored_points_sigma_0.18_dim_5.npy')
# nk['image'] = np.load(PATH  + '/NK/NK_image_anchored_points_sigma_0.18_dim_5.npy')

# mc['color'] = np.load(PATH + '/MC/MC_texture_color_anchored_points_sigma_0.18_dim_5.npy')

# sa['texture_grayscale'] = np.load(PATH + '/SA/SA_texture_grayscale_anchored_points_sigma_0.18_dim_5.npy')
# sa['intermediate_texture'] = np.load(PATH + '/SA/SA_intermediate_texture_anchored_points_sigma_0.18_dim_5.npy')
# sa['word'] = np.load(PATH + '/SA/SA_word_anchored_points_sigma_0.18_dim_5.npy')


# ranges = {'MC': {'texture': None, 'intermediate': None, 'word': None},
#            'SAW': {'texture': None, 'word': None},
#           'YCL': {'intermediate': None, 'word': None}
# }


def stretch_axes(subjects):
    """
    Run PCA.
    Stretch axes so as to make the variance along each axis the same.
    Do so by dividing the values of the coordinate by the standard deviation of values along that axis.
    This way points along each axis will have unit variance. This does not affect the radial
    distribution of points, only their distance from the origin in different directions.
    """
    n_components = 5
    pca = PCA(n_components=n_components)
    for sub in subjects:
        for exp in sub.keys():
            if exp not in ['texture', 'intermediate_texture', 'intermediate_object', 'image', 'word', 'color', 'color_old']:
                continue
            # obtain the 5 PC directions and project data onto that space
            print('Stretching axes for ' + str(exp))
            temp = pca.fit_transform(sub[exp])
            # normalize each axis by its standard deviation to make sd across each axis the same
            for i in range(n_components):
                temp[:, i] = temp[:, i]/np.std(temp[:, i])
            sub[exp] = temp


def violinplots_pairwise_distances():
    labels = []
    sns.set_style('dark')
    # sns.xkcd_palette(['purple', 'light purple', 'lavender', 'mocha', 'fawn', 'greyish', 'grey'])
    palette = {'word': 'purple', 'intermediate': 'gray', 'texture': 'brown'}
    pairwise_distances_df = {'subject': [], 'experiment': [], 'distances': []}
    vionlinplot_all = []
    for subject in [mc, ycl, saw]:
        for experiment in subject.keys():
            if experiment not in ['texture', 'intermediate', 'word']:
                continue
            pairwise_distances = pdist(subject[experiment])
            pairwise_distances = sorted(pairwise_distances)
            # should not normalize distances if I already normalized scatter across dimensions in PCA
            # pairwise_distances = [d / pairwise_distances[-1] for d in pairwise_distances]

            # add entry to dataframe
            for dist in pairwise_distances:
                pairwise_distances_df['subject'].append(subject['name'])
                pairwise_distances_df['experiment'].append(experiment)
                pairwise_distances_df['distances'].append(dist)

            labels.append(subject['name'] + ' \n'+ experiment )
            vionlinplot_all.append(list(pairwise_distances))
    violin_parts = plt.violinplot(vionlinplot_all, showmedians=True)
    plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7], labels=labels)
    # colors = ["#9b5fc0", "#7d7f7c", "#d3b683", "#9b5fc0", "#d3b683", "#9b5fc0", "#7d7f7c"] # set by hand
    color_dict = {"word": "#806E85", "intermediate": "#8E8D6A", "texture": "#31505A"}
    colors = [color_dict["word"], color_dict["texture"], color_dict["intermediate"],
              color_dict["word"], color_dict["intermediate"], color_dict["word"], color_dict["texture"]]
    for i in range(len(violin_parts['bodies'])):
        violin_parts['bodies'][i].set_color(colors[i])
        violin_parts['bodies'][i].set_edgecolor(colors[i])

    plt.ylabel('Normalized pairwise distances (37 points)')
    f = plt.gcf()
    f.savefig('/Users/suniyya/Desktop/violinplot1_all.png', dpi=200)
    plt.show()

    pairwise_distances_df = pd.DataFrame.from_dict(pairwise_distances_df)
    g = sns.violinplot(data=pairwise_distances_df[pairwise_distances_df['experiment'].isin(['word', 'texture']) & pairwise_distances_df['subject'].isin(['MC', 'SAW'])],
                   hue='experiment', split=True, y='distances', x='subject', inner="quartile",
                   # palette={"word": sns.xkcd_rgb["amethyst"], "texture": sns.xkcd_rgb["greyish"]})
                palette = {"word": "#806E85", "texture": "#31505A"})
    plt.legend(loc="upper center")
    plt.ylabel('Normalized Pairwise Distances')
    f = plt.gcf()
    f.savefig('/Users/suniyya/Desktop/violinplot2.png', dpi=200)
    plt.show()


    g = sns.violinplot(data=pairwise_distances_df[pairwise_distances_df['experiment'].isin(['word', 'intermediate']) & pairwise_distances_df['subject'].isin(['MC', 'YCL'])],
                   hue='experiment', split=True, y='distances', x='subject', inner='quartile',
                   palette={"word": "#806E85", "intermediate": "#8E8D6A"})
    plt.legend(loc="upper center")
    f = plt.gcf()
    f.savefig('/Users/suniyya/Desktop/violinplot3.png', dpi=200)
    plt.show()


    sns.swarmplot(data=pairwise_distances_df[pairwise_distances_df['experiment'].isin(['word', 'intermediate']) & pairwise_distances_df['subject'].isin(['MC', 'YCL'])],
                y='distances', x='subject', hue='experiment', dodge=True,
                palette={"word": sns.xkcd_rgb["amethyst"], "intermediate": sns.xkcd_rgb["pale brown"]}
    )
    plt.legend(loc="upper center")
    plt.ylabel('Normalized Pairwise Distances')
    f = plt.gcf()
    f.savefig('/Users/suniyya/Desktop/swarmplot1}.png', dpi=200)
    plt.show()


    sns.swarmplot(data=pairwise_distances_df[pairwise_distances_df['experiment'].isin(['word', 'texture']) & pairwise_distances_df['subject'].isin(['MC', 'SAW'])],
                y='distances', x='subject', hue='experiment', dodge=True,
                palette={"word": "C0", "texture": "C1"}
    )
    plt.ylabel('Normalized Pairwise Distances')
    plt.legend(loc="upper center")
    f = plt.gcf()
    f.savefig('/Users/suniyya/Desktop/swarmplot2.png', dpi=200)
    plt.show()



def scatterplots_3d():
    for subject in [mc, ycl, saw]:
        # no  pca
        for experiment  in subject.keys():
            if experiment not in ['texture', 'intermediate', 'word']:
                continue
            current_fig = plt.figure()
            ax = current_fig.add_subplot(111, projection='3d')
            ax.scatter(subject[experiment][:, 0], subject[experiment][:, 1], subject[experiment][:, 2], c='b', marker='o')
            ax.scatter([0], [0], [0], c='r', marker='x')
            ax.set_title(subject['name'] + ' ' + experiment)
            current_fig.savefig('/Users/suniyya/Desktop/scatterplot3d_{}_{}.png'.format(subject['name'], experiment), dpi=200)
            plt.show()

def scatterplots_2d():
    sns.set_style('darkgrid')
    colors = {"word": "#806E85", "intermediate_texture": "#31505A",
              "intermediate_object": "#8E8D6A", "image": "#8E8D6A", "texture": "#31505A", 'color': 'c'}
    stimuli = ['dolphin', 'goldfish', 'shark', 'whale', 'bluebird', 'duck', 'eagle', 'owl', 'pigeon', 'sparrow',
               'turkey',
               'mouse', 'rat', 'bat', 'bear', 'cat', 'dog', 'fox', 'goat', 'sheep', 'hog', 'cow', 'horse', 'giraffe',
               'elephant', 'monkey', 'tiger', 'ant', 'butterfly', 'ladybug', 'spider', 'snail', 'turtle', 'frog',
               'lizard',
               'snake', 'crocodile']
    for subject in [saw, mc, ycl]:
        # call stretch axes first
        for experiment  in subject.keys():
            if experiment not in ['texture',
                                  'intermediate_texture',
                                  'intermediate_object',
                                  'image',
                                  'word',
                                  'color']:
                continue
            current_fig = plt.figure()
            plt.scatter(subject[experiment][:, 0], subject[experiment][:, 1], c=colors[experiment], marker='.')
            # add labels to points
            label_idx = 0
            for x,y  in zip(subject[experiment][:, 0], subject[experiment][:, 1]):
                plt.annotate(stimuli[label_idx],  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 1.5),  # distance from text to points (x,y)
                             size=10,
                             ha='center')  # horizontal alignment can be left, right or center
                label_idx += 1

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(subject['name'] + ' ' + experiment)
            plt.axis('square')
            plt.ylim([-2, 3.5])
            plt.xlim([-2, 3.5])
            current_fig.savefig('/Users/suniyya/Desktop/scatterplot2d_{}_{}.png'.format(subject['name'], experiment), dpi=200)
            plt.show()


def scatterplots_3d_image_annotated(subject_name, subject_exp_data, image_source, pc1=1, pc2=2, pc3=3):
    stimuli = ['dolphin', 'goldfish', 'shark', 'whale', 'bluebird', 'duck', 'eagle', 'owl', 'pigeon', 'sparrow',
               'turkey',
               'mouse', 'rat', 'bat', 'bear', 'cat', 'dog', 'fox', 'goat', 'sheep', 'hog', 'cow', 'horse', 'giraffe',
               'elephant', 'monkey', 'tiger', 'ant', 'butterfly', 'ladybug', 'spider', 'snail', 'turtle', 'frog',
               'lizard',
               'snake', 'crocodile']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(subject_exp_data[:, pc1-1],
               subject_exp_data[:, pc2-1],
               subject_exp_data[:, pc3-1], c="#31505A", marker='.')
    # add labels to points
    label_idx = 0
    for x, y, z in zip(subject_exp_data[:, pc1-1], subject_exp_data[:, pc2-1], subject_exp_data[:, pc3-1]):
        if image_source is None:
            # for i in range(len(m)):  # plot each point + it's index as text above
            ax.text(x, y, z, '%s' % stimuli[label_idx], size=7, zorder=1,
                        color='k')
            # plt.annotate(stimuli[label_idx],  # this is the text
            #              (x, y, z),  # this is the point to label
            #              textcoords="offset points",  # how to position the text
            #              xytext=(0, 1.5),  # distance from text to points (x,y)
            #              size=10,
            #              ha='center')  # horizontal alignment can be left, right or center
            label_idx += 1
        else:
            filepath = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/stimulus_domains/images/{}/{}.png'\
                .format(image_source, stimuli[label_idx])
            with get_sample_data(filepath) as file:
                arr_img = plt.imread(file)
            imagebox = OffsetImage(arr_img, zoom=0.005)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (x, y, z),
                                xycoords = 'data',
                                boxcoords = "offset points",
                                pad = 0.1)
            ax.add_artist(ab)
            label_idx += 1

    ax.set_xlabel('Principal Component {}'.format(pc1))
    ax.set_ylabel('Principal Component {}'.format(pc2))
    ax.set_zlabel('Principal Component {}'.format(pc3))

    plt.title(subject_name)
    ax.set_xlim(-2, 3.5)
    ax.set_ylim(-2, 3.5)
    ax.set_zlim(-2, 3.5)

    plt.show()


def scatterplots_2d_image_annotated(subject_name, subject_exp_data, image_source, pc1=1, pc2=2):
    sns.set_style('darkgrid')
    stimuli = ['dolphin', 'goldfish', 'shark', 'whale', 'bluebird', 'duck', 'eagle', 'owl', 'pigeon', 'sparrow',
               'turkey',
               'mouse', 'rat', 'bat', 'bear', 'cat', 'dog', 'fox', 'goat', 'sheep', 'hog', 'cow', 'horse', 'giraffe',
               'elephant', 'monkey', 'tiger', 'ant', 'butterfly', 'ladybug', 'spider', 'snail', 'turtle', 'frog',
               'lizard',
               'snake', 'crocodile']

    fig, ax = plt.subplots()
    plt.scatter(subject_exp_data[:, pc1-1], subject_exp_data[:, pc2-1], c="#31505A", marker='.')
    # add labels to points
    label_idx = 0
    for x, y in zip(subject_exp_data[:, pc1-1], subject_exp_data[:, pc2-1]):
        if image_source is None:
            plt.annotate(stimuli[label_idx],  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 1.5),  # distance from text to points (x,y)
                         size=10,
                         ha='center')  # horizontal alignment can be left, right or center
            label_idx += 1
        else:
            filepath = '/Users/suniyya/Dropbox/Research/Thesis_Work/Psychophysics_Aim1/stimulus_domains/images/{}/{}.png'\
                .format(image_source, stimuli[label_idx])
            with get_sample_data(filepath) as file:
                arr_img = plt.imread(file)
            imagebox = OffsetImage(arr_img, zoom=0.005)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (x, y),
                                xycoords = 'data',
                                boxcoords = "offset points",
                                pad = 0.1)
            ax.add_artist(ab)
            label_idx += 1

    plt.xlabel('Principal Component {}'.format(pc1))
    plt.ylabel('Principal Component {}'.format(pc2))
    plt.title(subject_name)
    plt.axis('square')
    ax.set_xlim(-2, 3.5)
    ax.set_ylim(-2, 3.5)
    # ax.set_xlim(-0.4, 1.3)
    # ax.set_ylim(-0.8, 0.9)
    # current_fig.savefig('/Users/suniyya/Desktop/scatterplot2d_{}_{}.png'.format(subject['name'], experiment),
    #                     dpi=200)
    plt.show()


def other_plots():
    for subject in [mc, ycl, saw]:
        legend = []
        for experiment in subject.keys():
            if experiment not in ['texture', 'intermediate', 'word']:
                continue

            distances_from_origin = [np.linalg.norm(subject[experiment][i])
                                              for i in range(len(subject[experiment]))]
            distances_from_origin = sorted(distances_from_origin)
            # should not normalize distances if I already normalized scatter across dimensions in PCA
            # distances_from_origin = [d/distances_from_origin[-1] for d in distances_from_origin]
            legend.append(subject['name'] + ' ' + experiment)
            plt.plot(([0] + distances_from_origin), range(38), '.-')

        x = [ii for ii in range(10)]
        plt.xlabel('Normalized distance (radius) from origin (using 5D coordinates)')
        plt.ylabel('Number of points enclosed in a sphere of radius x')

        plt.legend(legend)
        plt.show()


    for subject in [mc, ycl, saw]:
        legend = []
        for experiment in subject.keys():
            if experiment not in ['texture', 'intermediate', 'word']:
                continue

            pairwise_distances = pdist(subject[experiment])
            pairwise_distances = sorted(pairwise_distances)
            # should not normalize distances if I already normalized scatter across dimensions in PCA
            # pairwise_distances = [d/pairwise_distances[-1] for d in pairwise_distances]
            legend.append(subject['name'] + ' ' + experiment)
            plt.plot(pairwise_distances, '.')

        plt.xlabel('')
        plt.ylabel('Normalized pairwise distance (using 5D coordinates, PCA, axis normalization)')
        legend.append('expected if points are regularly spaced')

        plt.legend(legend)
        plt.show()


    radii = [ii*0.1 for ii in range(60)]
    markers = {'texture': 'r-', 'intermediate': 'k-', 'word': 'b-'}
    for subject in [mc, ycl, saw]:
        legend = []
        for experiment in subject.keys():
            if experiment not in ['texture', 'intermediate', 'word']:
                continue
            distances_from_origin = [np.linalg.norm(subject[experiment][i])
                                              for i in range(len(subject[experiment]))]
            distances_from_origin = sorted(distances_from_origin)
            # should not normalize distances if I already normalized scatter across dimensions in PCA
            # distances_from_origin = [d/distances_from_origin[-1] for d in distances_from_origin]

            points_by_radius = [0]*len(radii)
            enclosed_points = [0]*len(radii)
            for d in distances_from_origin:
                for r in range(len(radii)):
                    if d <= radii[r]:
                        enclosed_points[r] += 1

            for idx in range(len(enclosed_points)):
                if idx > 0:
                    points_by_radius[idx] = enclosed_points[idx]-enclosed_points[idx-1]

            legend.append(subject['name'] + ' ' + experiment)
            plt.plot(radii, enclosed_points, markers[experiment], alpha=0.6)
            plt.xlabel('Normalized distance (radius) from origin (using 5D coordinates)')
            plt.ylabel('Number of points enclosed in a sphere of radius x')

        plt.legend(legend)
        plt.show()


    fig, axes = plt.subplots(1, 7, sharey=True)
    idx = 0
    for subject in [mc, ycl, saw]:
        for experiment in subject.keys():
            if experiment not in ['texture', 'intermediate', 'word']:
                continue
            distances_from_origin = [np.linalg.norm(subject[experiment][i])
                                              for i in range(len(subject[experiment]))]
            distances_from_origin = sorted(distances_from_origin)
            # should not normalize distances if I already normalized scatter across dimensions in PCA
            # distances_from_origin = [d / distances_from_origin[-1] for d in distances_from_origin]

            axes[idx].set_title(subject['name'] + ' ' + experiment)
            axes[idx].boxplot(distances_from_origin)
            idx += 1

    axes[0].set_ylabel('Normalized distance from the origin (37 points)')
    plt.show()




# for both models, repeat 10 times and get a curve with error bars or something

# randomly sample from a uniform distribution 37 points and calculate their distances from origin
def random_uniform_model(sample_size=1000, dim=5):
    # TODO
    points = np.array([[np.random.uniform(-3, 3) for _ in range(dim)] for _i in range(sample_size)])
    distances = pdist(points)
    distances = distances/ max(distances)
    points_gaussian = np.array([np.random.standard_normal(dim) for _i in range(sample_size)])
    distances_gaussian = pdist(points_gaussian)
    distances_gaussian = distances_gaussian/ max(distances_gaussian)

    points_intermediate = []
    for _ in range(sample_size):
        point = []
        while len(point) < dim:
            number = np.random.uniform(-3, 3)
            while not (number > 2 or number < -2):
                number = np.random.uniform(-3, 3)
            point.append(number)
        points_intermediate.append(point)

    points_skew = [skewnorm.rvs(-0.4, size=dim) for _ in range(sample_size)]
    distances_skew = pdist(points_skew)
    distances_skew = distances_skew/ max(distances_skew)

    points_beta = [np.random.beta(0.4, 0.4, dim) for _ in range(sample_size)]
    distances_beta = pdist(points_beta)
    distances_beta = distances_beta/max(distances_beta)
    distances_intermediate = pdist(points_intermediate)
    distances_intermediate = distances_intermediate/ max(distances_intermediate)
    plt.violinplot([list(distances), list(distances_gaussian), list(distances_intermediate), list(distances_beta),
                    list(distances_skew)], showmedians=True)
    labels = ['uniform dist', 'st. normal', 'uniform shell', 'beta dist', 'skew norm']
    plt.xticks([1, 2, 3, 4, 5], labels=labels)
    plt.show()



SUBJECTS = [saw]
stretch_axes(SUBJECTS)
for sub in SUBJECTS:
    # scatterplots_2d_image_annotated(sub['name'],
    #                                 sub['color'],
    #                                 'animal_textures/textures_color')
    scatterplots_2d_image_annotated(sub['name'],
                                    sub['image'],
                                    'animal_images')
    # scatterplots_3d_image_annotated(sub['name'],
    #                                 sub['word'],
    #                                 None)

# normalize by dividing by max distance.
# plot Num points by radius  - points evenly distributed uniformly distributed model
# points from a Gaussian model:
# randomly sample points from a Gaussian cloud
# rnorm along each dimension.
# similarly normalize by max distance and plot num points by radius

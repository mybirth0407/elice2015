import numpy
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import random
import sklearn.preprocessing

# Simple Linear Regression 2
def visualize(X, Y, results):

    def generate_random_permutation():
        return ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for i in range(16))

    def generate_base64_image(img_buffer):
        b64str = base64.b64encode(img_buffer.getvalue())
        permutation = generate_random_permutation()
        img_str = "<image %s>" % permutation
        img_str += str(b64str)[2:-1]
        img_str += "</%s>" % permutation
        return img_str

    slope = results.params[1]
    intercept = results.params[0]

    plt.scatter(X, Y)
    reg_line_x = numpy.array([min(X), max(X)])
    reg_line_y = reg_line_x * slope + intercept
    plt.plot(reg_line_x, reg_line_y, color='r')
    plt.show()

    format = "png"
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format=format)
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# Probability, PCA, Applying PCA, PCA + LOL, HDC, K-means Clustring, K-means vs DBScan 1 2 3, SVM, Clustring Stocks, PCA vs LDA
def generate_random_permutation():
    return ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for i in range(16))

def generate_base64_image(img_buffer):
    b64str = base64.b64encode(img_buffer.getvalue())
    permutation = generate_random_permutation()
    img_str = "<image %s>" % permutation
    img_str += str(b64str)[2:-1]
    img_str += "</%s>" % permutation
    return img_str

def visualize_boxplot(title, values, labels):
    width = .35

    fig, ax = plt.subplots()
    ind = numpy.arange(len(values))
    rects = ax.bar(ind, values, width)
    ax.bar(ind, values, width=width)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(labels)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., height + 0.01, '%.2lf%%' % (height * 100), ha='center', va='bottom')

    autolabel(rects)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# PCA
def draw_toy_example(df, pca, pca_array):
    plt.figure(figsize=(4.5, 4.5))

    X = np.array(df['x'].values)
    Y = np.array(df['y'].values)

    X = X - np.mean(X)
    Y = Y - np.mean(Y)

    line_X = np.arange(X.min() - 0.2, X.max() + 0.2, step=0.1)
    line_Y = (pca.components_[0, 1] / pca.components_[0, 0]) * line_X

    plt.ylim(min(min(line_X), min(line_Y)), max(max(line_X), max(line_Y)))
    plt.xlim(min(min(line_X), min(line_Y)), max(max(line_X), max(line_Y)))

    for x, y in zip(X, Y):
        plt.scatter(x, y)
    plt.plot(line_X, line_Y)

    pca_x = np.array(pca_array)
    pca_x = pca_x ** 2
    a = pca_x / (pca.components_[0, 1] ** 2 + pca.components_[0, 0] ** 2)
    a = np.sqrt(a)

    red_x = []
    red_y = []
    for i in range(0, len(a)):
        red_x.append(pca.components_[0, 0] * a[i] * np.sign(pca_array[i]))
        red_y.append(pca.components_[0, 1] * a[i] * np.sign(pca_array[i]))

    plt.scatter(red_x, red_y, c='r')

    for i in range(0, len(a)):
        plt.plot([X[i], red_x[i]], [Y[i], red_y[i]], ls='dotted', c='black')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# Applying PCA
def wine_graph(pca_array, class_df, class_names = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']):
    plt.figure(figsize=(6, 4.5))

    class_array = np.array(class_df)
    for c, i, class_name in zip("rgb", [1, 2, 3], class_names):
        plt.scatter(pca_array[class_array == i, 0], pca_array[class_array == i, 1], c=c, label=class_name, linewidth='0', s=6)

    plt.legend(loc=4)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# PCA + LOL, K-means Clustring
def plot_champions(champs_df, champ_pca_array):
    champ_names = champs_df.index.values

    x = champ_pca_array[:, 0]
    y = champ_pca_array[:, 1]
    difficulty = champs_df['difficulty'].values
    magic = champs_df['attack'].values

    plt.figure(figsize=(20, 10))

    plt.scatter(x, y,  c = magic, s = difficulty*1500, cmap = plt.get_cmap('Spectral'))

    for champ_name, x, y in zip(champ_names, x, y):
        plt.annotate(
            champ_name,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# K-means Clustring
def generate_random_permutation():
    return ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for i in range(16))

def generate_base64_image(img_buffer):
    b64str = base64.b64encode(img_buffer.getvalue())
    permutation = generate_random_permutation()
    img_str = "<image %s>" % permutation
    img_str += str(b64str)[2:-1]
    img_str += "</%s>" % permutation
    return img_str


# HDC
def display_digits(digits, index):
    plt.clf()
    plt.figure(1, figsize=(2, 2))
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)

def benchmark_plot(X, Y):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(X, Y, color='b', linestyle='dashed')
    ax.scatter(X, Y)
    ax.set_title("Benchmark: #Components from 1 to 64")
    ax.set_xlabel("#Principal Components")
    ax.set_ylabel("Homogeneity Score")

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)

def plot_digit_class(pca_array, num_classes):
    x = pca_array[:, 0]
    y = pca_array[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler()
    num_color = scaler.fit_transform(np.array(num_classes).astype('float64'))

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y,  c = num_color, s = 50, cmap = plt.get_cmap('Spectral'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# K-menas vs DBScan 1
def draw_init():
    plt.figure(figsize=(5, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)


def draw_graph(X, kmeans_result, alg_name, plot_num):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.subplot(2, 1, plot_num)
    if 1 == plot_num:
        plt.title(alg_name, size=18)
    plt.scatter(X[:, 0], X[:, 1], color=colors[kmeans_result.predict(X)].tolist(), s=10)

    centers = kmeans_result.cluster_centers_
    center_colors = colors[:len(centers)]
    plt.scatter(centers[:, 0], centers[:, 1], s=400, c=center_colors)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())


def show_graph():
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


def plot_digit_class(pca_array, num_classes):
    x = pca_array[:, 0]
    y = pca_array[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler()
    num_color = scaler.fit_transform(np.array(num_classes).astype('float64'))

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y,  c = num_color, s = 50, cmap = plt.get_cmap('Spectral'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# Kmeans vs DBScan 2
def draw_init():
    plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)


def draw_graph(X, algorithm, name, plot_num):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.subplot(2, 2, plot_num)
    if 2 >= plot_num:
        plt.title(name, size=18)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        center_colors = colors[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=400, c=center_colors)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())


def show_graph():
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


def plot_digit_class(pca_array, num_classes):
    x = pca_array[:, 0]
    y = pca_array[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler()
    num_color = scaler.fit_transform(np.array(num_classes).astype('float64'))

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y,  c = num_color, s = 50, cmap = plt.get_cmap('Spectral'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# K-means vs DBScan 3
def draw_init():
    plt.figure(figsize=(2 + 3, 2.5 * 9 + 0.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.99, wspace=.05, hspace=.1)


def draw_graph(X, dbscan_result, alg_name, plot_num, len_algs, indices):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.subplot(len_algs, 2, indices[plot_num-1])
    if len_algs >= plot_num:
        plt.title(alg_name, size=18)
    plt.scatter(X[:, 0], X[:, 1], color=colors[dbscan_result.labels_.astype(np.int)].tolist(), s=10)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())


def show_graph():
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


def plot_digit_class(pca_array, num_classes):
    x = pca_array[:, 0]
    y = pca_array[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler()
    num_color = scaler.fit_transform(np.array(num_classes).astype('float64'))

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y,  c = num_color, s = 50, cmap = plt.get_cmap('Spectral'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# SVM
def draw_init():
    plt.figure(figsize=(9, 9))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)


def draw_graph(X, y, svc_linear, svc_poly2, svc_poly3, svc_rbf, h = 0.2):
    draw_init()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    titles = ['SVC with linear kernel',
              'SVC with polynomial degree 2 kernel',
              'SVC with polynomial degree 3 kernel',
              'SVC with RBF kernel']

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    for i, clf in enumerate((svc_linear, svc_poly2, svc_poly3, svc_rbf)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        plt.scatter(X[:, 0], X[:, 1], color=colors[y.tolist()].tolist())
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    print(show_graph())

def show_graph():
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


def plot_digit_class(pca_array, num_classes):
    x = pca_array[:, 0]
    y = pca_array[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler()
    num_color = scaler.fit_transform(np.array(num_classes).astype('float64'))

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y,  c = num_color, s = 50, cmap = plt.get_cmap('Spectral'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# Clustring Stocks
def plot_stocks(df, pca_array, cluster_labels, code_to_name, display_cluster_idx):
    display_datapoints_indices = [i for i in range(0, len(cluster_labels)) if cluster_labels[i] == display_cluster_idx]

    names = df.index.values[display_datapoints_indices]

    x = pca_array[:, 0][display_datapoints_indices]
    y = pca_array[:, 1][display_datapoints_indices]

    scaler = sklearn.preprocessing.MinMaxScaler()
    colors = scaler.fit_transform(np.array(cluster_labels).astype('float64'))[display_datapoints_indices]

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, c = colors, cmap = plt.get_cmap('Spectral'))

    for name, x, y in zip(names, x, y):
        plt.annotate(
            code_to_name[name],
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


# PCA vs LDA
def draw_init():
    plt.figure(figsize=(5, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)


def draw_graph(X_pca, X_lda, y, svc_linear_pca, svc_rbf_pca, svc_linear_lda, svc_rbf_lda, h = 0.5):
    # title for the plots
    titles = ['Linear kernel SVC with PCA',
              'RBF kernel SVC with PCA',
              'Linear kernel SVC with LDA',
              'RBF kernel SVC with LDA']

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    for i, clf in enumerate((svc_linear_pca, svc_rbf_pca, svc_linear_lda, svc_rbf_lda)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        if i < 2:
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

            # Plot also the training points
            plt.scatter(X_pca[:, 0], X_pca[:, 1], color=colors[y.tolist()].tolist())

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())

        else:
            x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
            y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

            # Plot also the training points
            plt.scatter(X_lda[:, 0], X_lda[:, 1], color=colors[y.tolist()].tolist())

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
        plt.title(titles[i])


def show_graph():
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


def plot_digit_class(pca_array, num_classes):
    x = pca_array[:, 0]
    y = pca_array[:, 1]

    scaler = sklearn.preprocessing.MinMaxScaler()
    num_color = scaler.fit_transform(np.array(num_classes).astype('float64'))

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y,  c = num_color, s = 50, cmap = plt.get_cmap('Spectral'))

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    return generate_base64_image(img_buffer)


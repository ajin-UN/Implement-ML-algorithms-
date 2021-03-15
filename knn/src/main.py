from src.load_movielens import load_movielens_data
import numpy as np
import random

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


class KNearestNeighbor():
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        self.features = np.array([])
        self.targets = np.array([])

    def fit(self, features, targets):
        self.features = features
        self.targets = targets

    def predict(self, features, ignore_first=False):
        if self.distance_measure == 'euclidean':
            distances = euclidean_distances(features, self.features)
        elif self.distance_measure == 'manhattan':
            distances = manhattan_distances(features, self.features)
        else:
            distances = cosine_distances(features, self.features)

        prediction = []
        for distance in distances:
            indexes = distance.argsort()
            indexes = indexes[1: self.n_neighbors + 1] if ignore_first else indexes[: self.n_neighbors]
            train_targets = np.array(self.targets[indexes, :])

            cur_value = []
            for j in range(train_targets.shape[1]):
                if self.aggregator == 'mean':
                    cur_value.append(np.mean(train_targets[:, j]))

                elif self.aggregator == 'median':
                    cur_value.append(np.median(train_targets[:, j]))

                else:
                    dic = {}
                    for num in train_targets[:, j]:
                        if num in dic:
                            dic[num] += 1
                        else:
                            dic[num] = 1
                    max_num = -float("inf")
                    res = 0
                    for key in dic:
                        if dic[key] > max_num:
                            res = key
                            max_num = dic[key]
                    cur_value.append(float(res))
            prediction.append(cur_value)

        return prediction


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.
    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    print("enter euclidean_distances")
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])
    for i in range(0, M):
        print(i)
        for j in range(0, N):
            tmp = 0
            for k in range(0, K):
                tmp += (X[i, k] - Y[j, k]) ** 2
            D[i, j] = np.sqrt(tmp)
    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.
    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    print("manhattan")
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])

    for i in range(0, M):
        print(i)
        for j in range(0, N):
            tmp = 0
            for k in range(0, K):
                tmp += abs(X[i, k] - Y[j, k])
            D[i, j] = tmp
    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.
    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    print("enter cosine")
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    D = np.zeros([M, N])

    for i in range(0, M):
        print(i)
        for j in range(0, N):
            tmp = 0
            tmp1 = 0
            tmp2 = 0
            for k in range(0, K):
                tmp += (X[i, k] * Y[j, k])
                tmp1 += X[i, k] * X[i, k]
                tmp2 += Y[j, k] * Y[j, k]

            tmp1 = np.sqrt(tmp1)
            tmp2 = np.sqrt(tmp2)
            D[i, j] = 1 - (tmp / (tmp1 * tmp2))
            # D[i, j] = tmp / tmp1
    return D


def collaborative_filtering(input_array, n_neighbors,
                            distance_measure='euclidean', aggregator='mode'):
    print("enter collaborative_filtering")
    distance = 0

    # get distance
    if distance_measure == 'euclidean':
        distance = euclidean_distances(input_array, input_array)
    elif distance_measure == 'manhattan':
        distance = manhattan_distances(input_array, input_array)
    elif distance_measure == 'cosine':
        distance = cosine_distances(input_array, input_array)

    # get nearest n neighbors
    nearest = []

    for i in range(input_array.shape[0]):
        tmp = []
        for n in range(n_neighbors):
            smallest = 9999999999999999
            smallestIdx = 999999999999999
            for j in range(input_array.shape[0]):
                if distance[i, j] < smallest and (j not in tmp) and (i != j):
                    smallest = distance[i, j]
                    smallestIdx = j
            tmp.append(smallestIdx)
        nearest.append(tmp)

    # replace 0s
    # print()
    # print(input_array)
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            tmp = []
            if input_array[i, j] == 0:
                for k in nearest[i]:
                    # print(input_array[k, j])
                    tmp.append(input_array[k, j])
                # print(nearest[i])
                # print(tmp)
                t = 0
                if aggregator == 'mode':
                    t = max(set(tmp), key=tmp.count)
                    # print(t)
                elif aggregator == 'mean':
                    sumN = 0
                    for k in tmp:
                        sumN += k
                    t = sumN / len(tmp)

                elif aggregator == 'median':
                    tmp.sort()
                    if len(tmp) % 2 == 0:
                        t = (tmp[len(tmp) / 2] + tmp[len(tmp) / 2 - 1]) / 2
                    else:
                        t = tmp[int(len(tmp) / 2)]
                input_array[i, j] = t

    # print(input_array)
    return input_array


def free_res(N, K, D, A, rating, newOne, inD):
    print("enter free res")
    inputD = np.empty_like(rating)
    inputD[:] = rating
    if newOne == False:
        inputD = inD

    # N
    if newOne == True:
        for i in range(943):
            count = 0
            GG = 0
            while True:
                j = random.randint(0, 1681)
                GG += 1
                if inputD[i, j] != 0 and inputD[i, j] != 0:
                    inputD[i, j] = 0
                    count += 1
                    if count == N:
                        break
                if (GG == 1000):
                    print("GG")
                    break

    #
    if newOne == True:
        inputD = collaborative_filtering(inputD, K, D, A)

    # MSE
    MSE = 0
    counter = 0
    for i in range(943):
        for j in range(1682):
            if rating[i, j] != 0:
                MSE += (rating[i, j] - inputD[i, j]) ** 2
                counter += 1
    MSE /= counter
    MSE = np.sqrt(MSE)
    print("MSE")
    print(MSE)
    return MSE, inputD


def main():
    rating = load_movielens_data("data/ml-100k")
    # 8
    MSE1 = free_res(1, 3, 'euclidean', 'mean', rating, True,0)
    MSE2 = free_res(2, 3, 'euclidean', 'mean', rating, True,0)
    MSE4 = free_res(4, 3, 'euclidean', 'mean', rating, True,0)
    MSE8 = free_res(8, 3, 'euclidean', 'mean', rating, True,0)

    # 9
    # MSE_1 = free_res(1, 3, 'euclidean', 'mean', rating, True ,0)
    # MSE_2 = free_res(1, 3, 'cosine', 'mean', rating, True ,0)
    # MSE_3 = free_res(1, 3, 'manhattan', 'mean', rating, True, 0)

    # # 10
    # MSE1, aa = 0.2608256122392891, 0
    # MSE3, aa = free_res(1, 3, 'cosine', 'mean', rating, True, 0)
    # MSE7, aa = free_res(1, 7, 'cosine', 'mean', rating, True, 0)
    # MSE11, aa = free_res(1, 11, 'cosine', 'mean', rating, True, 0)
    # MSE15, aa = free_res(1, 15, 'cosine', 'mean', rating, True, 0)
    # MSE31, aa = free_res(1, 31, 'cosine', 'mean', rating, True, 0)

    # 11
    # MSE_1, aa = free_res(1, 7, 'cosine', 'mean', rating, True, 0)
    # MSE_2, aa = free_res(1, 7, 'cosine', 'mode', rating, False, aa)
    # MSE_3, aa = free_res(1, 7, 'cosine', 'median', rating, False, aa)
    # MSE_4, aa = free_res(1, 11, 'cosine', 'mean', rating, True, 0)
    # MSE_5, aa = free_res(1, 11, 'cosine', 'mode', rating, False, aa)
    # MSE_6, aa = free_res(1, 11, 'cosine', 'median', rating, False, aa)
    # MSE_7, aa = free_res(1, 15, 'cosine', 'mean', rating, True, 0)
    # MSE_8, aa = free_res(1, 15, 'cosine', 'mode', rating, False, aa)
    # MSE_9, aa = free_res(1, 15, 'cosine', 'median', rating, False, aa)
    # MSE_10, aa = free_res(1, 31, 'cosine', 'mean', rating, True, 0)
    # MSE_11, aa = free_res(1, 31, 'cosine', 'mode', rating, False, aa)
    # MSE_12, aa = free_res(1, 31, 'cosine', 'median', rating, False, aa)

    tmp = []
    x = np.random.uniform(-20, 20, size=(6))
    y = np.random.uniform(-20, 20, size=(6))
    x[0] = 1
    x[1] = 3
    x[2] = 7
    x[3] = 11
    x[4] = 15
    x[5] = 31
    y[0] = MSE1
    y[1] = MSE3
    y[2] = MSE7
    y[3] = MSE11
    y[4] = MSE15
    y[5] = MSE31

    # x = np.random.uniform(-20, 20, size=(12))
    # y = np.random.uniform(-20, 20, size=(12))
    # x[0] = 1
    # x[1] = 2
    # x[2] = 3
    # x[3] = 4
    # x[4] = 5
    # x[5] = 6
    # x[6] = 7
    # x[7] = 8
    # x[8] = 9
    # x[9] = 10
    # x[10] = 11
    # x[11] = 12
    # y[0] = MSE_1
    # y[1] = MSE_2
    # y[2] = MSE_3
    # y[3] = MSE_4
    # y[4] = MSE_5
    # y[5] = MSE_6
    # y[6] = MSE_7
    # y[7] = MSE_8
    # y[8] = MSE_9
    # y[9] = MSE_10
    # y[10] = MSE_11
    # y[11] = MSE_12

    plt.figure(figsize=(6, 4))
    plt.xlabel('value_of_N', fontsize=10)
    plt.ylabel('mean_squared_error', fontsize=10)
    plt.plot(x, y)
    plt.scatter(x,y)
    # plt.plot(, , label ='np.log10 test MSE')
    # plt.scatter(features[:, 0], features[:, 1], label='pred')
    plt.legend()
    # plt.savefig('question8' + ".png")
    plt.savefig('question8' + ".png")
    # plt.savefig('question10' + ".png")
    # plt.savefig('question11' + ".png")


if __name__ == "__main__":
    main()

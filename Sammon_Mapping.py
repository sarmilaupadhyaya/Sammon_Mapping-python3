import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import PCA

stoplist = set(stopwords.words('english'))


class SammonMapping():
    """
    :param magic_factor: factor by which new vactors should be updated
    :param threshold: error threshold
    :param new_vector: vector reduced to desired dimension using PCA.
    :param error: initialized error
    :param error_old: initial error initialized (hypothetical)
    :param distance_new_vector: distance between new vectors initially
    :param distance_old_vector: distance between old vectors
    :param difference_new_vector:difference between each new vector with each other
    :param a value which is the fraction of 1 and summation of all distance old vector
    """

    def __init__(self, vectors):
        """

        :param vectors: input vectors as numpy darray which is to be reduced to desired dimension.

        """
        self.vectors = vectors
        self.magic_factor = 0.4

        self.threshold = 1e-5
        self.new_vector = PCA(n_components=2).fit_transform(self.vectors)
        self.error = 1.0
        self.error_old = 2.0
        self.distance_old_vector = self.distance_vector(self.vectors)
        self.distance_new_vector = self.distance_vector(self.new_vector)
        self.difference_new_vector = self.subtract_vector()
        self.c = self.calculate_c()

    def error_calculate(self):
        """

        :return: error value of distance between all old vectors and distance between all new vectors
        """
        result = np.triu(
            np.nan_to_num((((self.distance_old_vector - self.distance_new_vector) ** 2) / self.distance_old_vector)),
            k=1)
        sum = result.sum()

        return sum * (1 / self.c)

    def calculate_c(self):
        """
        :return: calculated value of c
        """

        c = np.triu(self.distance_old_vector, k=1)
        sum = 0
        for i in range(self.distance_old_vector.shape[0]):
            another_sum = 0
            for x in c[i]:
                another_sum = another_sum + x
            sum += another_sum
        return sum

    def first_differentiate(self, index):
        """
         :param index: index of vector with respect to which first differentiation is to be done
        :return: a vector with desired value of first differentiation with respect to input vector.
        """

        first_term = np.nan_to_num(np.divide((self.distance_old_vector[index] - self.distance_new_vector[index]), (
            np.multiply(self.distance_old_vector[index], self.distance_new_vector[index]))))
        first_term_populated = np.array([first_term.T] * 2).T
        final_value = np.multiply(first_term_populated, self.difference_new_vector[index])
        summed = final_value.sum(axis=0)

        return (-2 * summed) / self.c

    def second_differentiate(self, index):
        """
        :param index: index of vector with respect to which second differentiation is to be done
        :return: a vector with desired value of second differentiation with respect to input vector.
        """

        first = self.distance_old_vector[index] - self.distance_new_vector[index]
        first_populated = np.array([(self.distance_old_vector[index] - self.distance_new_vector[index]).T] * 2).T
        second = np.array([np.multiply(self.distance_old_vector[index], self.distance_new_vector[index]).T] * 2).T
        third = np.nan_to_num(np.divide(np.square(self.difference_new_vector[index]),
                                        np.array([self.distance_new_vector[index].T] * 2).T))
        e = np.array([(1 + np.nan_to_num(np.divide(first, self.distance_new_vector[index]))).T] * 2).T
        ans = np.nan_to_num(np.divide((first_populated - np.multiply(third, e)), second))
        arg1 = ans.sum(axis=0)

        answer = (-2 * arg1) / self.c
        return answer

    def distance_vector(self, vector):
        """
        :param vector: ndarray which contains numbers of vectors and whose distance with each other is to be calculated.
        :return: a ndarray containing distance between every vectors with each other
        """

        d = (vector ** 2).sum(axis=-1)[:, np.newaxis] + (vector ** 2).sum(axis=-1)
        d -= 2 * np.squeeze(vector.dot(vector[..., np.newaxis]), axis=-1)
        d **= 0.5
        return d

    def subtract_vector(self):

        """
        :return: a n * n dimensional darray of numpy each containing difference between vectors of m dimension, \
        where m is desired dimension in our case
        """
        J, P = self.new_vector.shape
        empty_vector = np.zeros(shape=[J, J, P])

        for i, x in enumerate(self.new_vector):
            next_vector = np.array([x] * J)
            intermediate_vector = next_vector - self.new_vector
            empty_vector[i] = intermediate_vector

        return empty_vector

    def get_sammon_mapped_coordinate(self):
        """
        :return: desired vector with error minimized
        """
        while abs(self.error_old - self.error) > self.threshold:
            for index, new in enumerate(self.new_vector):
                first = self.first_differentiate(index)
                second = self.second_differentiate(index)
                delta = self.magic_factor * first / second

                self.new_vector[index] -= delta
                self.distance_new_vector = self.distance_vector(self.new_vector)
                self.difference_new_vector = self.subtract_vector()
            self.error_old = self.error
            self.error = self.error_calculate()
            print(self.error)
        return self.new_vector

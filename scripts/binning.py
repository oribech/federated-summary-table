import numbers
from dataclasses import dataclass, field
from functools import cached_property
from typing import Tuple, List, Union

import numpy as np


def lens(x, point):
    """
    Let k be the privacy requirement for min data points in
    a single bin and x a sort ascending sequence (vector).
    This checks if the number of values in x that <=point is at least k
    or zero.
    """
    arr = np.array(x)
    if point is None:
        point = -np.inf
    msk = arr > point
    return len(arr[msk]), len(arr[~msk])


@dataclass
class Binning:
    size: int = 10
    u: float = field(default_factory=lambda: np.random.uniform(0, 1, 1).item())

    @cached_property
    def k(self):
        return self.size - 1

    def bin2d(self, x1, x2) -> Tuple[List[float], List[float], List[float]]:
        f"""
        Let k be the privacy requirement for min data points in
        a single bin and x1,x2 sorted ascending sequences (vectors).
        This function takes two samples x1,x2 and creates a table with a single bins column
        And 2 additional columns for the frequencies of x1,x2 along the bins. The bins are 
        Calculated in a privacy preserving way by only releasing bins with frequency(1) >= k  or equal to 0.
        The bins boundaries defined as and ordered set specifying the open upper boundary of
        each bin -Inf=x0<x1<...<xn=Inf. Denote x_max(j) as the biggest point from x1,x2 points who fall
        in the J'th bin. and the x_min(j+1) as the smallest point from x1,x2 who fall in the J+1'th bins
        Then the J'th bin (upper) boundary is  x_max(j)*u+(1-u)*x_min(j+1) where us is sampled from u~U(0,1)
        For each bin creation This is done by the anonymize_boundry function a. The last bin boundary is inf.
        """
        # assert isinstance(x1, Iterable)
        # assert isinstance(x2, Iterable)
        x1, x2 = sorted(x1), sorted(x2)
        # Begin with empty lists
        bins = []
        frequencies1 = []
        frequencies2 = []
        for i in range(9999):
            # Get next bin (upper) boundary
            next_point = self.next_2d_point(x1, x2)
            # if next point is none it means that it is the end
            # print(f"next point=>: {next_point}")
            if next_point is None:
                if bins:
                    bins[-1] = np.inf
                    frequencies1[-1] += len(x1)
                    frequencies2[-1] += len(x2)
                return bins, frequencies1, frequencies2
            # f1 is the number of point in x1 till the next point
            # x1 is the values in x1 greater or equal to the next point.
            # And the same for x2,f2
            x1t, x2t = x1[:], x2[:]
            x1, f1 = self.clip_count(x1, next_point)
            x2, f2 = self.clip_count(x2, next_point)
            # Collect the bin and the frequencies, anonymize next point in order to preserve privacy
            bins.append(self.anonymize_boundary(next_point, x1, x2))
            frequencies1.append(f1)
            frequencies2.append(f2)
            assert len(bins) == len(frequencies1) == len(frequencies2)

    def anonymize_boundary(self, next_point, x1, x2):
        """
        Denote x_max(j) as the biggest point from x1,x2 points who fall
        in the J'th bin. and the x_min(j+1) as the smallest point from x1,x2 who fall in the J+1'th bins
        Then the J'th bin (upper) boundary is  x_max(j)*u+(1-u)*x_min(j+1) where us is sampled from u~U(0,1)
        For each bin creation.
        This function returns x_max(j)*u+(1-u)*x_min(j+1)
        :param next_point:
        :param x1:
        :param x2:
        :return:
        """
        smallest_point = self.smallest_point(x1, x2)
        if smallest_point is None:
            if self.u == 0:
                return next_point
            return None
        return self.u * smallest_point + (1 - self.u) * next_point

    def smallest_point(self, x1, x2) -> Union[None, float]:
        return self.min(
            x1[0] if not len(x1) == 0 else None,
            x2[0] if not len(x2) == 0 else None
        )

    def next_2d_point(self, x1, x2):
        """
        Let k be the privacy requirement for min data points in
        a single bin and x1,x2 sorted ascending sequences (vectors).
        This takes the next point value (starting from the right side)
        that leaves more or equal to k points or zero in each sequence behind it.
        This takes the next upper boundary from each sequence (vector) as if
        the binning was only on 1 sample. Out of the two points take the minimum
        if both are preserving privacy ((1) according to the definition) each one for its
        own sequence and for the other sequence. If not take the maximum.
        """
        q1 = self.next_1d_point(x1)
        q2 = self.next_1d_point(x2)
        if not (self.valid_point(x2, q2) and self.valid_point(x1, q1)):
            return None
        if self.valid_point(x2, q1) and self.valid_point(x1, q2):
            point = self.min(q1, q2)
            return point
        point = self.max(q1, q2)
        assert point is None or isinstance(point, numbers.Number)
        return point

    def valid_point(self, x, point):
        """
        Let k be the privacy requirement for min data points in
        a single bin and x a sort ascending sequence (vector).
        This checks if the number of values in x that <=point is at least k
        or zero.
        """
        arr = np.array(x)
        if point is None:
            point = -np.inf
        msk = arr > point
        return (not len(arr[msk]) in range(1, self.size)) and (not len(arr[~msk]) in range(1, self.size))

    def next_1d_point(self, x):
        """
        Let k be the privacy requirement for min data points in
        a single bin and x a sort ascending sequence (vector).
        This function outputs the next point from
        a vector x that has k points. If the number of points
        in the vector that >= k'th point is less
        k. Then the output is the largest (last) point.
        """
        if len(x) in range(self.size):
            return None
        if len(x[self.k:]) in range(0, self.size):
            return x[-1]
        return x[self.k]

    def clip_count(self, x, next_point):
        t = np.array(x)

        t = list(t[t > next_point])
        f = len(x) - len(t)
        assert isinstance(t, list)
        assert isinstance(f, numbers.Number)
        return t, f

    def max(self, a, b):
        if a is None and b is not None:
            return b
        if b is None and a is not None:
            return a
        if b is None and a is None:
            return None
        return max(a, b)

    def min(self, a, b):
        if a is None and b is not None:
            return b
        if b is None and a is not None:
            return a
        if b is None and a is None:
            return None
        return min(a, b)

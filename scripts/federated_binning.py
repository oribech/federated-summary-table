from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from scripts import utils
from scripts.binning import Binning

privacy_condition_number = 10


def get_name(**kwargs):
    return f'{kwargs=}'.split('=')[0]


def to_series(*args):
    """"""
    out = []
    for arg in args:
        assert isinstance(arg, Iterable), f"arg: {arg} is not a Sequence"
        out.append(pd.Series(arg))
    return tuple(out)


def reallocate(c, f2):
    f2 = pd.Series(f2)
    s = f2.sum()
    if s == 0:
        return np.repeat((1 / len(f2)) * c, len(f2))
    return list(c * f2 / s)


def normed(f2):
    f2 = pd.Series(f2)
    s = f2.sum()
    if s == 0:
        return pd.Series(np.repeat((1 / len(f2)), repeats=len(list(f2))))
    return f2 / s


def fix_student_frequencies(f_teacher, f_student):
    f_student_new = []
    # while len(f_student) > 0:
    for i in range(10 ** 5):
        assert i < 10 ** 5 - 1, "Not ending the loop too late"
        if not len(f_student) > 0:
            break
        f_student_subset = next_private_subset(f_student)
        subset_len = len(f_student_subset)
        f_teacher_subset = f_teacher[:subset_len]
        f_student_new.extend(list(sum(f_student_subset) * normed(f_teacher_subset)))
        f_student = f_student[subset_len:]
        f_teacher = f_teacher[subset_len:]
    return f_student_new


def valid(c):
    return c not in range(1, privacy_condition_number)


def next_private_subset(f_student):
    f_student = f_student[:]
    f_student_subset = []
    f_student_subset.append(f_student.pop(0))
    while (not valid(sum(f_student_subset))) and len(f_student) > 0:
        f_student_subset.append(f_student.pop(0))
    if not valid(sum(f_student)):
        f_student_subset.extend(f_student)
    return f_student_subset


def federated_binning(x1, x2, bins, f1, f2):
    x1, x2 = np.array(x1), np.array(x2)
    new_bins, f1_student, f1_teacher, f2_student, f2_teacher = [], [], [], [], []
    for bin_i in range(len(bins)):
        current_bin = bins[bin_i]
        f1_i = f1[bin_i]
        f2_i = f2[bin_i]
        current_x1, current_x2 = x1[x1 < current_bin], x2[x2 < current_bin]
        x1, x2 = x1[x1 >= current_bin], x2[x2 >= current_bin]
        new_sub_bins, new_f1, new_f2 = Binning().bin2d(list(current_x1), list(current_x2))
        if len(new_sub_bins) < 2:
            new_bins.append(current_bin)
            f1_student.append(len(current_x1))
            f2_student.append(len(current_x2))
            f1_teacher.append(f1_i)
            f2_teacher.append(f2_i)
        else:
            new_sub_bins[-1] = current_bin
            new_bins.extend(new_sub_bins)
            f1_student.extend(new_f1)
            f2_student.extend(new_f2)
            f1_teacher.extend(reallocate(f1_i, new_f1))
            f2_teacher.extend(reallocate(f2_i, new_f2))
    f1_student = fix_student_frequencies(f1_teacher, f1_student)
    f2_student = fix_student_frequencies(f2_teacher, f2_student)
    return new_bins, list(pd.Series(f1_student) + pd.Series(f1_teacher)), list(
        pd.Series(f2_student) + pd.Series(f2_teacher))


@dataclass
class TabularSummary:
    privacy_condition: int
    centers_data: List

    @property
    def tabular_summary(self):
        x1, x2 = self.centers_data[0]
        last_bin_max, first_bin_min = utils.get_last_first_bin(x1, x2, self.privacy_condition)
        bins, f1, f2 = Binning().bin2d(list(x1), list(x2))
        for center_i in range(1, len(self.centers_data)):
            x1, x2 = self.centers_data[center_i]
            last_bin, first_bin = utils.get_last_first_bin(x1, x2, self.privacy_condition)
            last_bin_max = max(last_bin_max, last_bin)
            first_bin_min = min(first_bin_min, first_bin)
            bins, f1, f2 = federated_binning(x1, x2, bins, f1, f2)
        return bins, f1, f2, last_bin_max, first_bin_min

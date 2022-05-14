import Levenshtein
import numpy as np

SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}


def is_sent_ok(sent, delimeters=SEQ_DELIMETERS):
    for del_val in delimeters.values():
        if del_val in sent and del_val != delimeters["tokens"]:
            return False
    return True


def check_split(source_token, target_tokens):
    if source_token.split("-") == target_tokens:
        return "$TRANSFORM_SPLIT_HYPHEN"
    else:
        return None
    
def apply_transformation(source_token, target_token):
    target_tokens = target_token.split()
    if len(target_tokens) > 1:
        transform = check_split(source_token, target_tokens)
        if transform:
            return transform
    return None


def perfect_align(t, T, insertions_allowed=0,
                  cost_function=Levenshtein.distance):
    # dp[i, j, k] is a minimal cost of matching first `i` tokens of `t` with
    # first `j` tokens of `T`, after making `k` insertions after last match of
    # token from `t`. In other words t[:i] aligned with T[:j].

    # Initialize with INFINITY (unknown)
    shape = (len(t) + 1, len(T) + 1, insertions_allowed + 1)
    dp = np.ones(shape, dtype=int) * int(1e9)
    come_from = np.ones(shape, dtype=int) * int(1e9)
    come_from_ins = np.ones(shape, dtype=int) * int(1e9)

    dp[0, 0, 0] = 0  # The only known starting point. Nothing matched to nothing.
    for i in range(len(t) + 1):  # Go inclusive
        for j in range(len(T) + 1):  # Go inclusive
            for q in range(insertions_allowed + 1):  # Go inclusive
                if i < len(t):
                    # Given matched sequence of t[:i] and T[:j], match token
                    # t[i] with following tokens T[j:k].
                    for k in range(j, len(T) + 1):
                        transform = \
                            apply_transformation(t[i], '   '.join(T[j:k]))
                        if transform:
                            cost = 0
                        else:
                            cost = cost_function(t[i], '   '.join(T[j:k]))
                        current = dp[i, j, q] + cost
                        if dp[i + 1, k, 0] > current:
                            dp[i + 1, k, 0] = current
                            come_from[i + 1, k, 0] = j
                            come_from_ins[i + 1, k, 0] = q
                if q < insertions_allowed:
                    # Given matched sequence of t[:i] and T[:j], create
                    # insertion with following tokens T[j:k].
                    for k in range(j, len(T) + 1):
                        cost = len('   '.join(T[j:k]))
                        current = dp[i, j, q] + cost
                        if dp[i, k, q + 1] > current:
                            dp[i, k, q + 1] = current
                            come_from[i, k, q + 1] = j
                            come_from_ins[i, k, q + 1] = q

    # Solution is in the dp[len(t), len(T), *]. Backtracking from there.
    alignment = []
    i = len(t)
    j = len(T)
    q = dp[i, j, :].argmin()
    while i > 0 or q > 0:
        is_insert = (come_from_ins[i, j, q] != q) and (q != 0)
        j, k, q = come_from[i, j, q], j, come_from_ins[i, j, q]
        if not is_insert:
            i -= 1

        if is_insert:
            alignment.append(['INSERT', T[j:k], (i, i)])
        else:
            alignment.append([f'CHANGE_{t[i]}', T[j:k], (i, i + 1)])

    assert j == 0

    return dp[len(t), len(T)].min(), list(reversed(alignment))
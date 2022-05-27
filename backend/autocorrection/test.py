import Levenshtein
import numpy as np

import re
from edlib import align, getNiceAlignment

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

def smoke_cigar(cigar, src=[], trg=[]):
  ans = []
  number = 0
  query_align = 0
  target_align = 0
  for c in cigar:
    if c.isdigit():
      number = number * 10
      number += int(c)
    else:
      if c == "=":
        ans.extend([(query_align + i, target_align + i) for i in range(number)])
        query_align  += number
        target_align += number
      elif c == "I":
        ans.append((-1, target_align))
        target_align += 1
      elif c == "D":
        ans.append((query_align, -1))
        query_align += 1
      else:
        ans.append((query_align, target_align))
        target_align += 1
        query_align += 1
      number = 0
  if src:
    assert len(ans) == len(src)

  return ans

TOKENIZER_REGEX = re.compile(r'(\W)')

def tokenizer(text):
    tokens = TOKENIZER_REGEX.split(text)
    return [t for t in tokens if len(t.strip()) > 0]

def getNiceAlignment(alignResult, query, target, gapSymbol="-"):
    target_pos = alignResult["locations"][0][0]
    if target_pos == None:
        target_pos = 0
    query_pos = 0  # 0-indexed
    target_aln = match_aln = query_aln = []
    cigar = alignResult["cigar"]
    tags = []
    toks = []
    trgs = []
    idx = []
    index = 0
    for num_occurrences, alignment_operation in re.findall("(\d+)(\D)", cigar):
        num_occurrences = int(num_occurrences)
        # print(num_occurrences, alignment_operation)
        if alignment_operation == "=":
            tar = target[target_pos: target_pos + num_occurrences]
            target_pos += num_occurrences
            que = query[query_pos: query_pos + num_occurrences]
            query_pos += num_occurrences
            for t, q in zip(tar, que):
                tags.append("KEEP")
                toks.append(t)
                trgs.append('')
                index += 1
            # match_aln += "|" * num_occurrences
        elif alignment_operation == "X":
            tar = target[target_pos: target_pos + num_occurrences]
            target_pos += num_occurrences
            que = query[query_pos: query_pos + num_occurrences]
            query_pos += num_occurrences
            for t, q in zip(tar, que):
                tags.append(f"CHANGE")
                toks.append(t)
                trgs.append(q)
                index += 1
            # match_aln += "." * num_occurrences
        elif alignment_operation == "D":
            tar = target[target_pos: target_pos + num_occurrences]
            target_pos += num_occurrences
            for t in tar:
                tags.append("DELETE")
                toks.append(t)
                trgs.append('')
                index += 1
        elif alignment_operation == "I":
            que = query[query_pos: query_pos + num_occurrences]
            query_pos += num_occurrences
            # if trgs[-1] == '':
            # trgs[-1] += toks[-1]
            for q in que:
                trgs[-1] += ' ' + q
            tags[-1] = "APPEND"
        else:
            raise Exception(
                "The CIGAR string from alignResult contains a symbol not '=', 'X', 'D', 'I'. Please check the validity of alignResult and alignResult.cigar")

    return tags, toks, trgs

def align_sentence(sent1, sent2):
    """from sent1 -> sent2"""
    s1 = sent1.split()
    s2 = sent2.split()

    return getNiceAlignment(align(s2, s1, task="path"), s2, s1)
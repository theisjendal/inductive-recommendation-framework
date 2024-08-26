import numpy as np
from loguru import logger
from scipy import stats


def average_precision(ranked_relevancy_list):
    """
    Calculates the average precision AP@k. In this setting, k is the length of
    ranked_relevancy_list.
    :param ranked_relevancy_list: A one-hot numpy list mapping the recommendations to
    1 if the recommendation was relevant, otherwise 0.
    :return: AP@k
    """

    if len(ranked_relevancy_list) == 0:
        a_p = 0.0
    else:
        p_at_k = ranked_relevancy_list * np.cumsum(ranked_relevancy_list, dtype=np.float32) / (1 + np.arange(ranked_relevancy_list.shape[0]))
        a_p = np.sum(p_at_k) / ranked_relevancy_list.shape[0]

    assert 0 <= a_p <= 1, a_p
    return a_p


def recall_at_k(relevance, k, max_recall=False, n_liked=None):
    n_liked = sum(relevance) if n_liked is None else n_liked
    return sum(relevance[:k]) / (n_liked if not max_recall else min(k, n_liked))


def precision_at_k(relevance, k):
    return sum(relevance[:k]) / k


def dcg(rank, n=10):
    r = np.zeros(n)
    if rank < n:
        r[rank] = 1

    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))


def tau_at_k(utility, k):
    # Cutoff must match length of utility list
    if len(utility) != k:
        return np.NaN

    tau, p = stats.kendalltau(utility[:k], sorted(utility, reverse=True)[:k])

    return tau


def ndcg_at_k(r, k, n_ground_truth):
    len_rank = k
    idcg_len = int(min(n_ground_truth, len_rank))

    # calculate idcg
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
    dcg = np.cumsum([1.0 / np.log2(idx + 2) if item else 0.0 for idx, item in enumerate(r)])[:len_rank]
    result = dcg / idcg
    return result.tolist()


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ser_at_k(ranked_labeled_items, top_pop_items, k, normalize=False):
    serendipitous_labeled_items = [
        (item, relevance)
        for item, relevance in ranked_labeled_items
        if item not in top_pop_items[:k]
    ]

    if l := len(serendipitous_labeled_items) == 0:
        return 0

    return sum([relevance for item, relevance in serendipitous_labeled_items]) / (
        l if normalize else 1
    )


def coverage(recommended_entities, recommendable_entities):
    return len(recommended_entities) / len(recommendable_entities)


def hr_at_k(relevance, cutoff):
    return 1 in relevance[:cutoff]


if __name__ == '__main__':
    logger.info(ndcg_at_k([0, 0, 1.0], 3))
    logger.info(ndcg_at_k([0, 0, 1], 3))

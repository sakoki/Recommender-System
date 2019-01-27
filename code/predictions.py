# Author: Koki Sasagawa
# Date: 10/8/2018

import numpy as np


def baseline(r_idx, m_idx, sparse_matrix):
    '''Predict ratings with overall ave, reviewer ave, and movie ave

    Rating calculated using the following equation:

    r = u + b_x + b_i

    Where:
    u = overall mean rating
    b_x = rating deviation of reviewer (ave rating of reviewer - u)
    b_i = rating deviation of movie (ave rating of movie - u)

    :params int r_idx: index of reviewer
    :params int m_idx: index of movie
    :params sparse_matrix: movie x reviewer matrix in sparse format
    :type sparse_matrix: scipy.sparse.csr.csr_matrix
    :returns prediction: predicted rating of target movie
    :rtype: float
    '''

    u = np.mean(sparse_matrix.data)
    b_x = np.mean(sparse_matrix[:,r_idx].data) - u
    b_i = np.mean(sparse_matrix[m_idx,:].data) - u
    prediction = u + b_x + b_i

    return prediction


def collaborative_filter(r_idx, m_idx, cos_sim, sparse_matrix, threshold):
    '''Predict rating using collaborative filtering

    Use known ratings a reviwer has left for other movies, filter
    with cosine similarity score above a certain threshold, and
    calculate the weighted average of their ratings.

    Depending on the threshold, some movies may not have any ratings to
    calculate a predicition. For these cases, return the baseline prediction
    calculated using the following equation:

    r = u + b_x + b_i

    Where:
    u = overall mean rating
    b_x = rating deviation of reviewer (ave rating of reviewer - u)
    b_i = rating deviation of movie (ave rating of movie - u)

    :params int r_idx: index of reviewer
    :params int m_idx: index of movie
    :params cos_sim: cosine similarity scores of movies
    :type cos_sim: scipy.sparse.csr.csr_matrix
    :params sparse_matrix: movie x reviewer matrix in sparse format
    :type sparse_matrix: scipy.sparse.csr.csr_matrix
    :params float threshold: value to filter similarity scores
    :returns prediction: predicted rating of target movie
    :rtype: float
    '''

    weighted_score_sum = 0
    sim_score_sum = 0

    # Get row vector containing all sim scores for select movie
    sim_score = cos_sim[m_idx,:]
    sim_score = zip(sim_score.indices, sim_score.data)

    # Get ratings from sparse matrix
    for idx, val in sim_score:
        if val >= threshold and idx != m_idx:
            rating = sparse_matrix[idx, r_idx]
            if rating > 0:
                weighted_score_sum += rating * val
                sim_score_sum += val

    if sim_score_sum > 0:
        prediction = weighted_score_sum / sim_score_sum
    else:
        prediction = baseline(r_idx, m_idx, sparse_matrix)

    return prediction


def collaborative_filter_predictions(data, col_idx_map, row_idx_map, cos_sim, sparse_matrix, threshold):
    '''Run collaborative filtering and generate rating predictions
    
    :params data: reviewer and movie pairs 
    :type data: DataFrame
    :params dict col_idx_map: maps reviewer ID with index
    :params dict row_idx_map: maps movie ID with index
    :params cos_sim: cosine similarity scores of movies
    :type cos_sim: scipy.sparse.csr.csr_matrix
    :params sparse_matrix: movie x reviewer matrix in sparse format
    :type sparse_matrix: scipy.sparse.csr.csr_matrix
    :params float threshold: value to filter similarity scores
    :returns prediction: predicted ratings
    :rtype: list of float
    '''
    
    predictions = []
    
    for i in data.itertuples():
        reviewer = i[1]
        movie = i[2]

        # Retrieve index of reviewer and movie
        reviewer = col_idx_map.get(reviewer, False)
        movie = row_idx_map.get(movie, False)

        # Make CF prediction if reviewer and movie exist in training set
        if reviewer and movie: 
            predictions.append(collaborative_filter(reviewer, movie, cos_sim, sparse_matrix, threshold))
        else:
            predictions.append(0)
    
    return predictions 
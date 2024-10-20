################################################################
# Machine Learning
# Programming Assignment - Bayesian Classifier
#
# bayes.py - functions for Bayesian classifier
#
# Author: R. Zanibbi
# Author: E. Lima
# Author: Brandon Ranallo
################################################################

import math
import numpy as np

from debug import *
from results_visualization import *

################################################################
# Cost matrices
################################################################

#
# initializes all weights to 1 except for correct classification
#
def uniform_cost_matrix( num_classes ):
    cost_matrix = np.ones((num_classes,num_classes))
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


# 
# initializes unequal cost matrix (I hope this is fine for unequal costs)
#
def bnrs_unequal_costs( num_classes ):
    # Rows: output class, Columns: Target (ground truth) class
    if num_classes == 2:
        return uniform_cost_matrix(num_classes=num_classes)

    else:
        return np.array([[-0.20, 0.07, 0.07, 0.07],
                     [0.07, -0.15, 0.07, 0.07],
                     [0.07, 0.07, -0.05, 0.07],
                     [0.03, 0.03, 0.03, 0.03]])


################################################################
# Bayesian parameters 
################################################################

#
# defining priors as the portion of data points of each class out of all data points
#
def priors( split_data ):

    est_priors = np.zeros(len(split_data))
    num_points = 0
    for label in split_data:
        num_points = num_points + len(label)

    for i in range(len(split_data)):
        est_priors[i] = float(len(split_data[i])) / num_points

    return est_priors

def bayesian_parameters( CLASS_DICT, split_data, title='' ):
    # Compute class priors, means, and covariances matrices WITH their inverses (as pairs)
    class_priors = priors(split_data)
    class_mean_vectors = list( map( mean_vector, split_data ) )
    class_cov_matrices = list( map( covariances, split_data ) )

    # Show parameters if title passed
    if title != '':
        print('>>> ' + title)
        show_for_classes(CLASS_DICT, "[ Priors ]", class_priors )

        show_for_classes(CLASS_DICT, "[ Mean Vectors ]", class_mean_vectors)
        show_for_classes(CLASS_DICT, '[ Covariances and Inverse Covariances]', class_cov_matrices )
        print('')

    return (class_priors, class_mean_vectors, class_cov_matrices)


################################################################
# Gaussians (for class-conditional density estimates) 
################################################################

def mean_vector( data_matrix ):
    # Axis 0 is along columns (default)
    return np.mean( data_matrix, axis=0)

#
# Takes in a matrix of datapoints with a common label and returns covariance matrix
# and inverse covariance matrix
#
def covariances( data_matrix ):
    # HEADS-UP: The product of the matrix by its inverse may not be identical to the identity matrix
    #           due to finite precision. Can use np.allclose() to test for 'closeness'
    #           to ideal identity matrix (e.g., np.eye(2) for 2D identity matrix)
    means = mean_vector( data_matrix )

    covariance = 0
    x_var = 0
    y_var = 0
    for data in data_matrix:
        covariance += (data[0] - means[0]) * (data[1] - means[1])
        x_var += (data[0] - means[0]) * (data[0] - means[0]) 
        y_var += (data[1] - means[1]) * (data[1] - means[1])
    
    covariance = covariance / len(data_matrix)
    x_var = x_var / len(data_matrix)
    y_var = y_var / len(data_matrix)
    
    covariance_matrix = np.array([[x_var, covariance],
                                  [covariance, y_var]])
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
    
    # Returns a pair: ( covariance_matrix, inverse_covariance_matrix )
    return ( covariance_matrix, inverse_covariance_matrix )


#
# Probability density function at the mean
#
def mean_density( cov_matrix ):
    det = np.linalg.det(cov_matrix)
    # Can discount exponent at the mean
    return 1.0 / ((2 * np.pi) * (det ** 0.5))

 

#
# returns an array of squared mahalanobis distances
#
def sq_mhlnbs_dist( data_matrix, mean_vector, cov_inverse ):
    # Square of distance from the mean in *standard deviations* 
    # (e.g., a sqared mahalanobis distance of 9 implies a point is sqrt(9) = 3 standard
    # deviations from the mean.

    # Numpy 'broadcasting' insures that the mean vector is subtracted row-wise
    
    square_distances = np.zeros((len(data_matrix)))

    for x in range(len(data_matrix)):
        diff = data_matrix[x] - mean_vector
        square_distances[x] = np.matmul(np.matmul(np.transpose(diff), cov_inverse), diff)

    return square_distances

def gaussian( mean_density, distances ):
    # NOTE: distances is a column vector of squared mahalanobis distances

    # Use numpy matrix op to apply exp to all elements of a vector
    scale_factor = np.exp( -0.5 * distances )

    # Returns Gaussian values as the value at the mean scaled by the distance
    return mean_density * scale_factor


################################################################
# Bayesian classification
################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#     ** Where indicated
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def map_classifier( priors, mean_vectors, covariance_pairs ):
    # Unpack data once outside definition (to avoid re-computation)
    covariances =  np.array( [ cov_pair[0] for cov_pair in covariance_pairs ] )
    peak_scores = priors * np.array( [ mean_density(c) for c in covariances ] )

    inv_covariances =  np.array( [ cov_pair[1] for cov_pair in covariance_pairs ] )

    dnpcheck('covariances shape: ', inv_covariances.shape)

    num_classes = len(priors)

    def classifier( data_matrix ):
        num_samples = data_matrix.shape[0]

        # Create arrays to hold distances and class scores
        distances = np.zeros( ( num_samples, num_classes ) )
        class_scores = np.zeros( ( num_samples, num_classes + 1 ) ) 

        #>>>>>>>>> EDIT THIS SECTION
        
        for c in range(num_classes):
            mhlnbs_distances = sq_mhlnbs_dist(data_matrix=data_matrix, mean_vector=mean_vectors[c], cov_inverse=inv_covariances[c])
            scores = gaussian(peak_scores[c], distances=distances[:, c])
            for x in range(num_samples):
                distances[x, c] = mhlnbs_distances[x]

                class_scores[x, c] = scores[x]

        class_scores[:, num_classes] = np.argmax(class_scores[:, :-1], axis=1)
        
        #>>>>>>>>>> END SECTION TO EDIT
        
        return class_scores

    return classifier


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> REWRITE THIS FUNCTION
#     ** Where indicated
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def bayes_classifier( cost_matrix, priors, mean_vectors, covariance_pairs ):
    # Unpack data once outside definition (to avoid re-computation)
    covariances =  np.array( [ cov_pair[0] for cov_pair in covariance_pairs ] )
    peak_scores = priors * np.array( [ mean_density(c) for c in covariances ] )

    inv_covariances =  np.array( [ cov_pair[1] for cov_pair in covariance_pairs ] )
    num_classes = len(priors)

    def classifier( data_matrix ):
        num_samples = data_matrix.shape[0]

        # Create arrays to hold distances and class scores
        distances = np.zeros( ( num_samples, num_classes ) )
        class_posteriors = np.zeros( ( num_samples, num_classes ) ) 
        class_costs_output = np.zeros( ( num_samples, num_classes + 1) ) 

        #>>>>>>>>> EDIT THIS SECTION
        
        class_costs_output[:,-1] = np.ones( num_samples )
        
        #>>>>>>>>>> END SECTION TO EDIT

        return class_costs_output

    return classifier




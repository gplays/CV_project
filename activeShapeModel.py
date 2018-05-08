
class ActiveShapeModel(object):

    def __init__(self,centroid,tooth_id):
        # Set up pca information for this tooth
        # Mean_shape
        # Eig_val
        # Eig_vector
        pass

    def update_centroid(self):
        pass

    def update_landmarks(self):
        pass


    def deform_mean(self):
        # This function should deform the mean landmark shape using the eigen
        #  vectors. x = 'x' + E b Where b are the factor. Be sure to keep the
        #  factor in b smaller  than |3 sqrt(eig_val)| to keep closeness
        # The deformation might occur in a normalised space, be sure to
        # denormalize it after hand
        pass

    def compute_mean_error(self):
        # Compute l2 regularization error regarding b parameters
        # First need to regularize b with eig_vals
        pass

    def compute_intensity(self):
        # Compute cumulated intensity of landmarks
        # Help to align on lines
        pass

    def compute_fitness(self):
        # Combine mean_error and intenisty
        pass

    def eval_model(self):
        # Compare model with ground truth: Jaccard coeff on area
        pass

    def std_transform(self):
        # Apply standardisation mean and scaling in both dimensions
        pass

    def std_reverse(self):
        # Unapply standardisation mean and scaling in both dimensions
        pass
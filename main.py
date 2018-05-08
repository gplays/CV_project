from data_loader import RadioGraph
import pca_analysis
import utils.preprocess_image as pre
import cv2

if __name__ == "__main__":

    img = cv2.imread('./.data/Radiographs/02.tif')
    pre.process_image_test(img)

    # radiographs = [RadioGraph("./.data", i) for i in range(1, 15)]
    # pca_analysis.run_pca(radiographs)
    ## Part 1: Build an Active Shape Model
    # We used the description fromthe original paper by Cootes et al.


    # Load the provided landmarks into your program

    # Preprocess  the  landmarks  to  normalize  translation,  rotation  and
    # scale differences (Procrustes Analysis?)

    # Analyze the data using a Principal Component Analysis (PCA), exposing 
    # shape class variations

    # Analyze the obtained principal components

    ## Part 2 : Preprocess dental radiographs

    ## Part 3 : Fit the model to the image

    # First Guess

    # Iterative fitting


    pass

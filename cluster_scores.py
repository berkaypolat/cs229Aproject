import numpy as np

def read_similarity_matrix(filename):
    with open(filename, "r") as f:
        if f.mode != "r":
            raise FileNotFoundError("File could not be read.")

        sim_matrix = []
        lines = f.readlines()
        for line in lines:
            sim_row = []
            for num in line.split(","):
                sim_row.append(float(num))
            sim_matrix.append(sim_row)
        return np.array(sim_matrix)


"""
Commented out lines can be used to calculate the average score per each
clustering algorithm. Current version computes the direct sum of each score of 
each article per algorithm
"""
def compute_scores(CSMatrix):
    n_articles = 1224
    n_algorithms = 12
    matrix = read_similarity_matrix(CSMatrix).reshape(n_articles,n_articles,n_algorithms)
    clusterings = ['spectral_' + str(i) for i in range(1,11)]
    clusterings.append('dbscan')
    clusterings.append('affinity')
    dct_scores = {title:0 for title in clusterings}

    for z in range(matrix.shape[2]):
        slice_matrix = matrix[:,:,z]
        score = 0
        for col in range(slice_matrix.shape[1]):
            #non_zero_count = 0
            #temp_score = 0
            for row in range(slice_matrix.shape[0]):
                if(slice_matrix[row,col] != 0):
                    #non_zero_count += 1
                    #temp_score += slice_matrix[row,col]
                    score += slice_matrix[row,col]
            #avg_temp_score = temp_score / non_zero_count
            #score += avg_temp_score

        #total_avg_score = score / n_articles
        if (z < 10):
            #dct_scores['spectral_' + str(z+1)] = total_avg_score
            dct_scores['spectral_' + str(z+1)] = score
        elif (z == 10):
            #dct_scores['dbscan'] = total_avg_score
            dct_scores['dbscan'] = score
        else:
            #dct_scores['affinity'] = total_avg_score
            dct_scores['affinity'] = score
    
    return dct_scores
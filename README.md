# plagiarism-checker
Takes a set of documents, compares each with other to generate similarity measures - these could be cosine, jaccard, manhattan, etc. We tried to take 12 of them and shove them through a classifier such as SVM to learn the classification.

feature_list = ['cosine_similarity', 'euclidean_distance', 'E_jaccard', 'Pearson_Correlation', 'dice', 'manhattan_distance','jaccard_similarity', 'Bray_Curtis_Distance', 'Canberra_Distance','Chebyshev Distance', 'hamming', 'overlap', 'Minkowski Distance']

import pandas as pd                                               # importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel
from matplotlib import pyplot as plt

entertainment = pd.read_csv("Entertainment.csv")                  # importing datasets

# data cleansing and eda part
entertainment .shape 
entertainment .columns
entertainment .Category 
entertainment .isnull().sum() 
entertainment .info()

# univariate anlysis
plt.hist(entertainment.Reviews); plt.xlabel("reviews");plt.ylabel("count")

# bivariate analysis
plt.scatter(entertainment.Id,entertainment.Reviews); plt.xlabel("Id");plt.ylabel("reviews")

# TFIDF initialised and transformed
tfidf = TfidfVectorizer(stop_words = "english")
tfidf_matrix = tfidf.fit_transform(entertainment .Category )   
tfidf_matrix.shape 

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
entertainment_index = pd.Series(entertainment.index, index = entertainment['Titles']).drop_duplicates()
entertainment_index.duplicated().sum()
entertainment_id = entertainment_index["Heat (1995)"]
entertainment_id

# user define model for recommendation    
def get_recommendations(Name, topN):     
      entertainment_id = entertainment_index[Name]
      cosine_scores = list(enumerate(cosine_sim_matrix[entertainment_id]))
      cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
      cosine_scores_N = cosine_scores[0: topN+1]
      entertainment_idx  =  [i[0] for i in cosine_scores_N]
      entertainment_scores =  [i[1] for i in cosine_scores_N]
      
      
      entertainment_similar_show = pd.DataFrame(columns=["name", "Score"])
      entertainment_similar_show["name"] = entertainment.loc[entertainment_idx, "Titles"]
      entertainment_similar_show["Score"] = entertainment_scores
      entertainment_similar_show.reset_index(inplace = True)  
    
      print (entertainment_similar_show)
      
# passing the values to function for recommendations
get_recommendations("Lamerica (1994)", topN = 10)
   
      
a = ["aashish","anurag","dheeraj"]      
for idx,name in enumerate(a):
     print(idx,name)       
    
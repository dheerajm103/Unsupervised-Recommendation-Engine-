import pandas as pd                                               # importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel
from matplotlib import pyplot as plt

G= pd.read_csv("game.csv")                                        # importing datasets              

# data cleansing and eda part
G.shape 
G.columns
G.game.isnull().sum() 
G.info()

# univariate anlysis
plt.hist(G.rating); plt.xlabel("rating");plt.ylabel("count")
plt.bar(G.game,G.rating); plt.xlabel("game");plt.ylabel("rating")
# bivariate analysis
plt.scatter(G.userId,G.rating); plt.xlabel("userid");plt.ylabel("rating")
plt.scatter(G.game,G.rating); plt.xlabel("game");plt.ylabel("rating")

# initialising and fittig tfidf
tfidf = TfidfVectorizer(stop_words = "english")
tfidf_matrix = tfidf.fit_transform(G.game )   
tfidf_matrix.shape 

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
G_index = pd.Series(G.index, index = G['userId']).drop_duplicates()

# checking index
G_id = G_index[14]
G_id

# creating user define model for recommendation
def get_recommendations(userid, topN):     
      
      G_id = G_index[userid]
      cosine_scores = list(enumerate(cosine_sim_matrix[G_id]))
      cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
      cosine_scores_N = cosine_scores[0: topN+1]
      G_idx  =  [i[0] for i in cosine_scores_N]
      G_scores =  [i[1] for i in cosine_scores_N]
      
      
      G_similar_show = pd.DataFrame(columns=["userid","game", "Score"])
      G_similar_show["userid"] = G.loc[G_idx, "userId"]
      G_similar_show["Score"] =G_scores
      G_similar_show["game"] = G.loc[G_idx, "game"]
      G_similar_show.reset_index(inplace = True)  
    
      print (G_similar_show)
      
# passing the variables to function to get recommendation
get_recommendations(14, topN = 10)          
      
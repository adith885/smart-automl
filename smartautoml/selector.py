from sklearn.feature_selection import SelectKBest, f_classif

def get_selector():
    return SelectKBest(score_func=f_classif, k="all")
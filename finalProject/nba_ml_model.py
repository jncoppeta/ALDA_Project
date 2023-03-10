#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn import tree
from sklearn.svm import SVC
from sklearn.tree import (
    export_text,
)  # you can use this to display the tree in text formats
from sklearn.metrics import accuracy_score

# from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


START_YEAR, END_YEAR = 1973, 2004

# parameters for decisions tree will be criterion of gini and entropy
parameters_dt = {"criterion": ["gini", "entropy"]}
# parameters for knn will be n_neighbors from 3 to 10 and different distance metrics, p value 1 to 4
parameters_knn = {
    "n_neighbors": range(3, 11),
    "metric": ["minkowski"],
    "p": range(1, 5),
}
# the most optimal was .1 , I tried .1, .2. .5, and 1 but they take too long to run
parameters_linear_svc = {
    "C": [0.1],
    "kernel": ["linear"],
}

parameters_kfcv = {
    'clf__n_estimators':[100, 200, 500],
    'clf__max_depth': [5, 6, 7, 8]
}

# parameters_poly_svc = {
#     "C": [0.1, 1],
#     "degree": [2, 3, 4],
#     "coef0": [ 0.001, 0.01, 0.1, 1, 5, 10],
#     "kernel": ["poly"],
# }

# parameters_rbf_svc = {
#     "C": [0.1, 1],
#     "gamma": [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3],
#     "kernel": ["rbf"],
# }

# parameters_sigmoid_svc = {
#     "C": [0.1, 1],
#     "gamma": [0.0001, 0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3],
#     "coef0": [ 0.001, 0.01, 0.1, 1, 5, 10],
#     "kernel": ["sigmoid"],
# }


# map the years to all the players in that year
years_and_players = {}
years_and_players_cleaned = {}

# data in allstar txt file that is not needed
irrelevant_allstar_data = "conference,leag,gp,minutes,pts,dreb,oreb,reb,asts,stl,blk,turnover,pf,fga,fgm,fta,ftm,tpa,tpm".split(
    ","
)

# ilkid and year ?
aggregated_dimensions_player_data = (
    "team,leag,firstname,lastname,fga,fgm,fta,ftm,tpa,tpm".split(",")
)

aggregated_allstar_dimensions = "firstname,lastname".split(",")

# Sensitivity (TPR or recall) = True Positive / True Positive + False Negative this is all correct predictions over all the all stars
# this is better than accuracy because correct/all predictions will result in high accuracy given a bad model
def main():
    # DATA CLEANING:
    # ilkid,year,firstname,lastname,team,leag,gp,minutes,pts,oreb,dreb,reb,asts,stl,blk,turnover,pf (needed?),(fga (attempted),fgm (made)),(fta (free throw att),ftm),(tpa,tpm)
    # attempts and made we can make percentages to reduce our dimensions **

    print("\n===ORIGINAL REGULAR SEASON DATA===: \n\n")

    # I could not get this to work on my end so I commented it out so you can still run it when you want. Line 82 is the call I use for the same thing.
    # player_regular_season_data = pd.read_csv("./data/player_regular_season.txt")

    player_regular_season_data = pd.read_csv(r"C:\Users\jncop\OneDrive\Documents\GitHub\engr-ALDA-Fall2022-H3/finalProject/data/player_regular_season.txt")
    print(player_regular_season_data)

    print(
        "\n===DIMENSIONALITY REDUCTION ON REGULAR SEASON DATA and REMOVE DATA BEFORE 73===: \n\n"
    )
    player_regular_season_data["field_goal_percentage"] = (
        player_regular_season_data["fgm"] / player_regular_season_data["fga"]
    ) * 100
    player_regular_season_data["field_throw_percentage"] = (
        player_regular_season_data["ftm"] / player_regular_season_data["fta"]
    ) * 100
    player_regular_season_data["three_point_percentage"] = (
        player_regular_season_data["tpm"] / player_regular_season_data["tpa"]
    ) * 100
    # if field_goal_percentage is nan, then replace with 0
    player_regular_season_data["field_goal_percentage"] = player_regular_season_data["field_goal_percentage"].fillna(0)
    player_regular_season_data["field_throw_percentage"] = player_regular_season_data["field_throw_percentage"].fillna(0)
    player_regular_season_data["three_point_percentage"] = player_regular_season_data["three_point_percentage"].fillna(0)
    player_regular_season_data["id"] = (
        player_regular_season_data["ilkid"]
        + "-"
        + player_regular_season_data["year"].astype(str)
    )
    iterableRows = player_regular_season_data.iterrows()
    for index, player in iterableRows:
        if player["year"] < START_YEAR or player["year"] > END_YEAR:
            player_regular_season_data = player_regular_season_data.drop(index)
            # i = player_regular_season_data[((player_regular_season_data.ilkid == player["ilkid"])
            # & (player_regular_season_data.year == player["year"]))].index
    for dim in aggregated_dimensions_player_data:
        player_regular_season_data = player_regular_season_data.drop(dim, axis=1)
    print(player_regular_season_data)

    # ilkid,year,firstname,lastname (maybe combine ilkid concat with year for single identification player)
    print("\n===ALLSTAR SEASON DATA===: \n\n")

    # player_allstar_data = pd.read_csv("./data/player_allstar.txt")
    player_allstar_data = pd.read_csv(r"C:\Users\jncop\OneDrive\Documents\GitHub\engr-ALDA-Fall2022-H3/finalProject/data/player_allstar.txt")
    
    print(player_allstar_data)
    print("\n===DIMENSIONALITY REDUCTION ON ALLSTAR DATA===: \n\n")
    for x in irrelevant_allstar_data:
        player_allstar_data = player_allstar_data.drop(x, axis=1)

    player_allstar_data["ilkid"] = player_allstar_data["ilkid"].str.upper()
    player_allstar_data["ilkid"] = player_allstar_data["ilkid"].str.strip()
    player_allstar_data["y"] = (
        player_allstar_data["ilkid"] + "-" + player_allstar_data["year"].astype(str)
    )
    for dim in aggregated_allstar_dimensions:
        player_allstar_data = player_allstar_data.drop(dim, axis=1)
    print(player_allstar_data)

    # combine ilkid+year, drop firstname and lastname, all other data but add 0 or 1 where 0=non-allstar that season and 1=allstar that season.
    # get a set to put all of the player_allstar_data y values into
    allstars = set()
    for index, player in player_allstar_data.iterrows():
        allstars.add(player["y"])
    # create a new column in player data filled with all zeros unless the set contains the id then put 1
    player_regular_season_data["isAllStar"] = 0
    for index, player in player_regular_season_data.iterrows():
        if str(player["id"]) in allstars:
            player_regular_season_data.loc[index, "isAllStar"] = 1
    # Weed out bench players and non-contributors:
    #    1> player must have at least 55 games played
    player_regular_season_data_cleaned = player_regular_season_data.copy()
    for index, player in player_regular_season_data_cleaned.iterrows():
        if int(player["gp"]) < 55:
            player_regular_season_data_cleaned = (
                player_regular_season_data_cleaned.drop(index)
            )
            # i = player_regular_season_data_cleaned[((player_regular_season_data_cleaned.ilkid == player["ilkid"])
            #                                         & (player_regular_season_data_cleaned.year == player["year"]))]

    print(
        f"Total players considered for training/testing: {len(player_regular_season_data)}"
    )
    print(
        f"Total players considered training/testing *cleaned*: {len(player_regular_season_data_cleaned)}"
    )

    for i in range(START_YEAR, END_YEAR + 1):
        years_and_players[str(i)] = set()
        years_and_players_cleaned[str(i)] = set()

    for _, player in player_regular_season_data.iterrows():
        years_and_players[str(player["year"])].add(str(player["ilkid"]))
    for _, player in player_regular_season_data_cleaned.iterrows():
        years_and_players_cleaned[str(player["year"])].add(str(player["ilkid"]))

    data = player_regular_season_data.copy()
    data = data.drop("ilkid", axis=1)
    data = data.drop("id", axis=1)
    data_55gp = player_regular_season_data_cleaned.copy()
    data_55gp = data_55gp.drop("ilkid", axis=1)
    data_55gp = data_55gp.drop("id", axis=1)

    print("\n\n===data used for models===\n")
    print(data)
    print("Data with at least 55 games played:\n")
    print(data_55gp)

     #  SPLIT: 3-fold cross validation
    # use a 10 year seasons
    # 73-79, 80-86, 87-93
    print("===3-fold cross validation===")
    splitSet = set()

    for yearStr, players in years_and_players.items():
        year = int(yearStr)
        for player in players:
            if year >= 1973 and year <= 2004:
                splitSet.add(player + "-" + yearStr)

    print(f"size for 1974-2004: {len(splitSet)}")
    

    cleanedSetOne = set()
    cleanedSetTwo = set()
    cleanedSetThree = set()
    cleanedSetFour = set()
    for yearStr, players in years_and_players_cleaned.items():
        year = int(yearStr)
        for player in players:
            if len(cleanedSetOne) < 1866:
                cleanedSetOne.add(player + "-" + yearStr)
            elif len(cleanedSetTwo) < 1866:
                cleanedSetTwo.add(player + "-" + yearStr)
            elif len(cleanedSetThree) < 1866:
                cleanedSetThree.add(player + "-" + yearStr)
            elif len(cleanedSetFour) < 1866:
                cleanedSetFour.add(player + "-" + yearStr)
    print(f"size for cleanedSetOne: {len(cleanedSetOne)}")
    print(f"size for cleanedSetTwo: {len(cleanedSetTwo)}")
    print(f"size for cleanedSetThree: {len(cleanedSetThree)}")
    print(f"size for cleanedSetFour: {len(cleanedSetFour)}")
    print(f"total number of seasons: {len(cleanedSetOne) + len(cleanedSetTwo) + len(cleanedSetThree) + len(cleanedSetFour)}")


    allStarSet1 = set()
    allStarSet2 = set()
    allStarSet3 = set()
    allStarSet4 = set()
    allStars = set()
   
    for x in player_allstar_data["y"]:
        year = int(x[len(x) - 4:len(x)])
        if year >= 1973 and year <= 2004:
            allStars.add(x)
    print(len(allStars))
    for x in allStars:
        if len(allStarSet1) < 201:
            allStarSet1.add(x)
        elif len(allStarSet2) < 201:
            allStarSet2.add(x)
        elif len(allStarSet3) < 201:
            allStarSet3.add(x)
        elif len(allStarSet4) < 201:
            allStarSet4.add(x)
    print(f"\nsize for first All-Star set: {len(allStarSet1)}")
    print(f"size for second All-Star set: {len(allStarSet1)}")
    print(f"size for third All-Star set: {len(allStarSet1)}")
    print(f"size for four All-Star set: {len(allStarSet1)}")
    print(len(allStarSet1)+len(allStarSet2)+len(allStarSet3)+len(allStarSet4))

     # Classifier array of all 11864 seasons 1973-2004. 
    length = 11864
    kfClassifier = ["0"]*length
    playerArray = list(splitSet)
    for i in range(length - 1):
        x = list(splitSet)[i]
        if x in allStars:
            kfClassifier[i] = "1"

    foldOneX = set()
    foldOneY = set()
    for x in cleanedSetOne.union(cleanedSetTwo.union(cleanedSetThree)):
        if( x not in allStarSet1.union(allStarSet2.union(allStarSet3)) ):
            foldOneX.add(x)
    for x in cleanedSetFour:
        if( x not in allStarSet4 ):
            foldOneY.add(x)

    xTrain = []
    yTrain = []
    xTest = []
    yTest = []
    for x in foldOneX:
        xTrain.append(x)
    for x in allStarSet1.union(allStarSet2.union(allStarSet3)):
        yTrain.append(x)
    for x in foldOneY:
        xTest.append(x)
    for x in allStarSet4:
        yTest.append(x)
    print(len(xTrain))
    print(len(yTrain))
    


    rows, cols = ((END_YEAR + 1) - START_YEAR), 5
    rawDataResults = [[0]*cols]*rows
    eligibleDataResults = [[0]*cols]*rows
    averageCVAccuracy = 0
    averageTestAccuracy = 0
    averagePrecision = 0
    averageRecall = 0
    averageFScore = 0
    # CROSS-VALIDATION:
    #   SPLIT: LOOCV
    #     one season as TEST set and all other seasons as TRAINING set
    #        Measure sensitivity for each and average at end
    print("\n\n===LEAVE ONE OUT CROSS-VALIDATION===\n")
    for i in range(START_YEAR, END_YEAR + 1):
        # print(f"current season as TEST set: {i}")
        # get all of the players with matching year to str(i) as test set
        test_set = data[data["year"].astype(int) == i]
        test_set_55gp = data_55gp[data_55gp["year"].astype(int) == i]
        # get all other players as training set
        training_set = data[data["year"].astype(int) != i]
        training_set_55gp = data_55gp[data_55gp["year"].astype(int) != i]
        # now drop all the years from all data since its not needed other than identifying the player
        training_set = training_set.drop("year", axis=1)
        test_set = test_set.drop("year", axis=1)
        training_set_55gp = training_set_55gp.drop("year", axis=1)
        test_set_55gp = test_set_55gp.drop("year", axis=1)
        # now split the training set into X and y
        X_train = training_set.drop("isAllStar", axis=1)
        y_train = training_set["isAllStar"]
        X_test = test_set.drop("isAllStar", axis=1)
        y_test = test_set["isAllStar"]
        # now split the training set into X and y
        X_train_55gp = training_set_55gp.drop("isAllStar", axis=1)
        y_train_55gp = training_set_55gp["isAllStar"]
        X_test_55gp = test_set_55gp.drop("isAllStar", axis=1)
        y_test_55gp = test_set_55gp["isAllStar"]
        # now run the model on the training set and test it on the test set
        # three models: ID3 tree, Support Vector Machine, and KNN
        # ID3 tree
  
        print(f"\n{i} ", end="")
        # print("\n\n===10-Fold Cross Validation===:")
        # print("Regular Season Data:")
        svm = SVC()
        raw =  report_metrics(svm, parameters_linear_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data with at least 55 games played:")
        svm = SVC()
        eligibleResults = report_metrics(svm, parameters_linear_svc, X_train_55gp, y_train_55gp, X_test_55gp, y_test_55gp)
        for k in eligibleResults:
            print('%.f' % eligibleResults[k], end = "")
            print(" ", end = "")
        #averageTestAccuracy += eligibleResults[0]
        #averagePrecision += eligibleResults[1]
        #averageRecall += eligibleResults[2]
        #averageFScore += eligibleResults[3]
            
    #averageTestAccuracy = averageTestAccuracy/((END_YEAR + 1) - START_YEAR)
    #averagePrecision = averagePrecision/((END_YEAR + 1) - START_YEAR)
    #averageRecall = averageRecall/((END_YEAR + 1) - START_YEAR)
    #averageFScore = averageFScore/((END_YEAR + 1) - START_YEAR)
    print('\n\n                                               AVERAGES')
    #print('                     %.6f     %.6f     %.6f     %.6f     %.6f' % (averageCVAccuracy, averageTestAccuracy, averagePrecision, averageRecall, averageFScore))






    # CROSS-VALIDATION:
    #   SPLIT: LOOCV
    #     one season as TEST set and all other seasons as TRAINING set
    #        Measure sensitivity for each and average at end
    print("\n\n===LEAVE ONE OUT CROSS-VALIDATION===\n")
    for i in range(START_YEAR, END_YEAR + 1):
        print(f"current season as TEST set: {i}")
        # get all of the players with matching year to str(i) as test set
        test_set = data[data["year"].astype(int) == i]
        test_set_55gp = data_55gp[data_55gp["year"].astype(int) == i]
        # get all other players as training set
        training_set = data[data["year"].astype(int) != i]
        training_set_55gp = data_55gp[data_55gp["year"].astype(int) != i]
        # now drop all the years from all data since its not needed other than identifying the player
        training_set = training_set.drop("year", axis=1)
        test_set = test_set.drop("year", axis=1)
        training_set_55gp = training_set_55gp.drop("year", axis=1)
        test_set_55gp = test_set_55gp.drop("year", axis=1)
        # now split the training set into X and y
        X_train = training_set.drop("isAllStar", axis=1)
        y_train = training_set["isAllStar"]
        X_test = test_set.drop("isAllStar", axis=1)
        y_test = test_set["isAllStar"]
        # now split the training set into X and y
        X_train_55gp = training_set_55gp.drop("isAllStar", axis=1)
        y_train_55gp = training_set_55gp["isAllStar"]
        X_test_55gp = test_set_55gp.drop("isAllStar", axis=1)
        y_test_55gp = test_set_55gp["isAllStar"]
        # now run the model on the training set and test it on the test set
        # three models: ID3 tree, Support Vector Machine, and KNN
        # ID3 tree

        rows, cols = ((END_YEAR + 1) - START_YEAR), 4
        rawDataResults = [[0]*cols]*rows
        eligibleDataResults = [[0]*cols]*rows
        averageCVAccuracy = 0
        averageTestAccuracy = 0
        averagePrecision = 0
        averageRecall = 0
        averageFScore = 0
        for i in range(START_YEAR, END_YEAR + 1):
            print(f"\n{i}", end="")
            print("          ", end="")
            # print("\n\n===10-Fold Cross Validation===:")
            # print("Regular Season Data:")
            id3 = DecisionTreeClassifier()
            raw = report_metrics(id3, parameters_dt, X_train, y_train, X_test, y_test)
            # print("Regular Season Data with at least 55 games played:")
            id3 = DecisionTreeClassifier()
            eligibleResults = report_metrics(id3, parameters_dt, X_train_55gp, y_train_55gp, X_test_55gp, y_test_55gp)
            for k in range(cols):
                print('%.6f' % raw[k], end = "")
                print("          ", end = "")
            averageTestAccuracy += raw[0]
            averagePrecision += raw[1]
            averageRecall += raw[2]
            averageFScore += raw[3]
        averageTestAccuracy = averageTestAccuracy/((END_YEAR + 1) - START_YEAR)
        averagePrecision = averagePrecision/((END_YEAR + 1) - START_YEAR)
        averageRecall = averageRecall/((END_YEAR + 1) - START_YEAR)
        averageFScore = averageFScore/((END_YEAR + 1) - START_YEAR)
        print('\n\n                                               AVERAGES')
        print('                     %.6f     %.6f     %.6f     %.6f     %.6f' % (averageCVAccuracy, averageTestAccuracy, averagePrecision, averageRecall, averageFScore))
        
        
        dataFrameX = pd.DataFrame(kfClassifier)
        dataFrameY = pd.DataFrame(playerArray)
        # print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(trainDataY.iloc[train]), score))
        print("\n\n===ID3 Tree===:")
        id3 = DecisionTreeClassifier()
        print("Regular Season Data:")
        report_metrics(id3, parameters_dt, X_train, y_train, X_test, y_test)
        print("Regular Season Data with at least 55 games played:")
        id3 = DecisionTreeClassifier()
        report_metrics(
            id3, parameters_dt, X_train_55gp, y_train_55gp, X_test_55gp, y_test_55gp
        )
        # KNN
        print("\n===KNN===:")
        knn = KNeighborsClassifier()
        print("Regular Season Data:")
        report_metrics(knn, parameters_knn, X_train, y_train, X_test, y_test)
        print("Regular Season Data with at least 55 games played:")
        knn = KNeighborsClassifier()
        report_metrics(
            knn, parameters_knn, X_train_55gp, y_train_55gp, X_test_55gp, y_test_55gp
        )
        # Support Vector Machine
        print("\n===Support Vector Machine===:")
        print("Regular Season Data (linear):")
        svm = SVC()
        # report_metrics(svm, parameters_linear_svc, X_train, y_train, X_test, y_test)
        print("Regular Season Data with at least 55 games played (linear):")
        svm = SVC()
        report_metrics(
            svm,
            parameters_linear_svc,
            X_train_55gp,
            y_train_55gp,
            X_test_55gp,
            y_test_55gp,
        )
        
        # print("Regular Season Data (polynomial):")
        # svm = SVC()
        # report_metrics(svm, parameters_poly_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data (Radial Basic Function):")
        # svm = SVC()
        # report_metrics(svm, parameters_rbf_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data (sigmoid):")
        # svm = SVC()
        # report_metrics(svm, parameters_sigmoid_svc, X_train, y_train, X_test, y_test)
        # print("Regular Season Data with at least 55 games played (polynomial):")
        # svm = SVC()
        # report_metrics(
        #     svm,
        #     parameters_poly_svc,
        #     X_train_55gp,
        #     y_train_55gp,
        #     X_test_55gp,
        #     y_test_55gp,
        # )
        # print(
        #     "Regular Season Data with at least 55 games played (Radial Basic Function):"
        # )
        # svm = SVC()
        # report_metrics(
        #     svm,
        #     parameters_rbf_svc,
        #     X_train_55gp,
        #     y_train_55gp,
        #     X_test_55gp,
        #     y_test_55gp,
        # )
        # print("Regular Season Data with at least 55 games played (sigmoid):")
        # svm = SVC()
        # report_metrics(
        #     svm,
        #     parameters_sigmoid_svc,
        #     X_train_55gp,
        #     y_train_55gp,
        #     X_test_55gp,
        #     y_test_55gp,
        # )
        print("=====================================================")

   

def report_cv(trainDataX, trainDataY, testDataX, testDataY,folds):
    try:
        returnArr = [0]*5
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=None))
        model = KFold(n_splits=folds, shuffle=True, random_state=None)
        kfold = model.split(trainDataX, trainDataY)
        scores = []
        for k, (train, test) in enumerate(kfold):
            pipeline.fit(trainDataX.iloc[train, :], trainDataY.iloc[train])
            score = pipeline.score(trainDataX.iloc[test, :], trainDataY.iloc[test])
            scores.append(score)
            # print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(trainDataY.iloc[train]), score))
        predicted =  pipeline.predict(testDataX)
        # print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

        #print(f"==Accuracy Score: {accuracy_score(testDataY, predicted)}")
        #print(
        #    f"==Precision Score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[0]}"
        #)
        #print(
        #    f"==Recall Score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[1]}"
        #)
        #print(
        #    f"==F-score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[2]}"
        #)
        returnArr[0] = np.mean(scores)
        returnArr[1] = accuracy_score(testDataY, predicted)
        returnArr[2] = precision_recall_fscore_support(testDataY, predicted, average='binary')[0]
        returnArr[3] = precision_recall_fscore_support(testDataY, predicted, average='binary')[1]
        returnArr[4] = precision_recall_fscore_support(testDataY, predicted, average='binary')[2]
        return returnArr
    except ValueError as e:
            print("*SKIPPING THIS MODEL* ValueError: ", e)

def report_metrics(model, parameters, trainDataX, trainDataY, testDataX, testDataY):
    try:
        returnArr = [0]*5
        model = GridSearchCV(model, parameters, cv=None)
        model.fit(trainDataX, trainDataY)
        predicted = model.predict(testDataX)
        return predicted
        #print(f"==Accuracy Score: {accuracy_score(testDataY, predicted)}")
        #print(
        #    f"==Precision Score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[0]}"
        #)
        #print(
        #    f"==Recall Score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[1]}"
        #)
        #print(
        #     f"==F-score: {precision_recall_fscore_support(testDataY, predicted, average='binary')[2]}"
        #)
        #print(f"==Best parameters: {model.best_params_}")
        returnArr[0] = accuracy_score(testDataY, predicted)
        returnArr[1] = precision_recall_fscore_support(testDataY, predicted, average='binary')[0]
        returnArr[2] = precision_recall_fscore_support(testDataY, predicted, average='binary')[1]
        returnArr[3] = precision_recall_fscore_support(testDataY, predicted, average='binary')[2]
        returnArr[4] = model.best_params_
        return returnArr
    except ValueError as e:
            print("*SKIPPING THIS MODEL* ValueError: ", e)


main()

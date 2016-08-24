import sys
import numpy as np
import pandas as pd
import time
current_milli_time = lambda: int(round(time.time() * 1000))

from sklearn.ensemble import RandomForestClassifier

# train_path = sys.argv[1]
# train_truth_path = sys.argv[2]

print("train_df")
train_df = np.genfromtxt('../features', delimiter=',')

print("train_truth_df")
truth_df = np.genfromtxt('../truth', delimiter=',')

train_df = train_df.reshape((2247, 442))
truth_df = truth_df.reshape((2247, 1))[:,0]

rfc = RandomForestClassifier(n_estimators=10)

for _ in range(0, 10):
    msk = np.random.rand(2247) < 0.9
    train_feat = train_df[msk]
    train_truth = truth_df[msk]
    print("train_feat: {}", train_feat.shape)

    print("train_truth: {}", train_truth.shape)
    pred_feat = train_df[~msk]
    print("pred_feat: {}", pred_feat.shape)
    # pred_truth = truth_df[~msg]

    start = current_milli_time()
    rfc.fit(train_feat, train_truth)
    end = current_milli_time()

    print("Time to fit: {}", end - start)

    start = current_milli_time()
    rfc.predict(pred_feat)
    end = current_milli_time()

    print("Time to predict: {}", end - start)



# pred = rfc.predict(test_df)
#
# score = rfc.score(test_df, test_truth_df)
#
# for index in range(0, len(test_truth_df)):
#     if pred[index] != test_truth_df[index]:
#         print(index)
#         print("{} {} {}".format(prob[index], pred[index], test_truth_df[index]))
#
# print(score)

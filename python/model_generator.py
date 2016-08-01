import sys
import numpy as np
import pandas as pd
import time
current_milli_time = lambda: int(round(time.time() * 1000))

from sklearn.ensemble import RandomForestClassifier

train_path = sys.argv[1]
train_truth_path = sys.argv[2]
test_path = sys.argv[3]
test_truth_path = sys.argv[4]
print("train_df")
train_df = np.genfromtxt(train_path, delimiter=',')

print("train_truth_df")
train_truth_df = np.genfromtxt(train_truth_path, delimiter=',')


print("test_df")
test_df = np.genfromtxt(test_path, delimiter=',')

print("test_truth_df")
test_truth_df = np.genfromtxt(test_truth_path, delimiter=',')

print(train_df.shape)
print(train_truth_df.shape)
print(test_df.shape)
print(test_truth_df.shape)


rfc = RandomForestClassifier(n_estimators=25)
start = current_milli_time()
rfc.fit(train_df, train_truth_df)
end = current_milli_time()

print("Time to fit: {}", end - start)

prob = rfc.predict_proba(test_df)[:, 1]


pred = rfc.predict(test_df)

score = rfc.score(test_df, test_truth_df)

for index in range(0, len(test_truth_df)):
    print("{} {} {}".format(prob[index], pred[index], test_truth_df[index]))

print(score)

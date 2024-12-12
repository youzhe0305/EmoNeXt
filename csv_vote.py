import pandas as pd
from collections import Counter

file_paths = ["output1.csv", "output2.csv", "output3.csv", "output4.csv", "output5.csv",
              "output6.csv", "output7.csv", "output8.csv", "output9.csv", "output10.csv",
              "output11.csv", "output12.csv"]

priority_file_index = 8

dfs = [pd.read_csv(file) for file in file_paths]

merged_df = pd.concat(dfs, keys=range(len(dfs)), names=['source', 'index']).reset_index(level=0)
merged_df = merged_df.pivot_table(index='filename', columns='source', values='label', aggfunc='first').reset_index()

def vote(row):
    labels = row.drop('filename').values
    counter = Counter(labels)
    most_common = counter.most_common()

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # print('row:')
        # print(len(row))
        # print(row)
        priority_label = row[priority_file_index]
        return priority_label
    else:
        return most_common[0][0]

merged_df['final_label'] = merged_df.apply(vote, axis=1)

result_df = merged_df[['filename', 'final_label']]
result_df.to_csv("output_vote_12.csv", index=False)

print("voting complete, result has been saved to output_vote.csv")

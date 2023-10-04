import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import sys

class KNN:
  def __init__(self, k):
    self.k = k
    self.attrs = None # Features
    self.y_classifs = None  # Class labels
    self.classifs = None   # Unique class labels

  def fit(self, X, y):
    self.attrs = X
    self.y_classifs = y
    self.classifs = self.y_classifs.T[0]

  def predict(self, value, weighted=False):
    # Calculate distances
    dist = np.sqrt(np.sum((self.attrs - value)**2, axis=1).astype(np.float64))
    # Sort indeces of distances
    sorted_idxs = np.argsort(dist)

    # If distance is 0, return class label with highest count
    if weighted and 0.0 in dist:
      return np.bincount(self.classifs[np.where(dist == 0.0)[0]]).argmax()

    # Find neighbours to get predicted class
    weights = self.analyze_neighbours(dist, sorted_idxs, weighted, self.k)

    # If there are ties, increase k until there's no more ties
    sum = 1
    while(len(weights) != len(set(weights))):
      weights = self.analyze_neighbours(dist, sorted_idxs, weighted, self.k + sum)
      sum += 1

    # Return class label with highest weight or count
    return max(weights, key = weights.get)


  def analyze_neighbours(self, dist, sorted_idxs, weighted, k):
    # Find k-nearest neighbours and calculate weights if weighted mode is enabled
    weights = {}
    for i in sorted_idxs[:k]:
      if self.classifs[i] not in weights:
        weights[self.classifs[i]] = 0
      weights[self.classifs[i]] += (1/dist[i]) if weighted else 1
    return weights

def partition_data(df, test_p):
  # Shuffle rows
  df = df.sample(frac=1).reset_index(drop=True)

  # Get partition size
  partition_size = int(np.floor(len(df) * test_p))
  partitions = []

  end = 0
  start = partition_size
  while end < len(df):
    partitions.append(df[end:start].copy())
    end += partition_size
    start += partition_size
    if(start > len(df)):
      start = len(df)

  if(start - end) != partition_size:
    partitions[-2] = pd.concat([partitions[-2], partitions[-1]], ignore_index=True)
    partitions = partitions[:-1]

  return partitions


def get_avg_wordcount(df, rating):
  df_rated = df[df['Star Rating'] == rating]
  word_count = df_rated['wordcount'].sum()
  print("Average word count for reviews rated", rating)
  print(word_count / len(df_rated))


def knn_classification(df, k, test_p, weighted, remove_na, normalize):
  df = df[['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']]
  if remove_na:
    df = df.dropna()
  else:
    # Attempt to fill NaN with approximate values
    df.loc[:, 'titleSentiment'] = df.loc[:, 'titleSentiment'].fillna(0)
    df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] >= 3), 'titleSentiment'] = 'positive'
    df.loc[(df['titleSentiment'] == 0) & (df['Star Rating'] < 3), 'titleSentiment'] = 'negative'

  df.loc[df['titleSentiment'] == 'positive', 'titleSentiment'] = 1
  df.loc[df['titleSentiment'] == 'negative', 'titleSentiment'] = 0

  # Normalize values
  if(normalize):
    scaler = MinMaxScaler()
    df.loc[:, ['wordcount', 'titleSentiment', 'sentimentValue']] = scaler.fit_transform(df[['wordcount', 'titleSentiment', 'sentimentValue']].values)

  partitions = partition_data(df, test_p)

  knn = KNN(k)

  i = 0
  entry = {}
  for p in partitions:
    test = p
    train = pd.concat([df for df in partitions if df is not p])
    train.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    knn.fit(train[['wordcount', 'titleSentiment', 'sentimentValue']].to_numpy(), train[['Star Rating']].to_numpy())

    df_tested = pd.DataFrame(columns = ['predicted', 'real'])
    for inst in test.to_numpy():
      atts = inst[0:-1]
      y = inst[-1]

      prediction = knn.predict(atts, weighted)
      entry['predicted'] = prediction
      entry['real'] = int(y)
      df_tested = pd.concat([df_tested, pd.DataFrame([entry])], ignore_index=True)

    df_tested.to_csv("classification" + str(i) + ".csv")
    i+=1

def calculate_matrix(df, confusion_matrix=None):
    correct = 0
    for _, row in df.iterrows():
        predicted = row['predicted']
        real = row['real']
        if predicted == real:
            correct += 1

        if confusion_matrix is not None:
            confusion_matrix[real][predicted] += 1

    return correct

def get_matrix_heatmap(df):
  plt.clf()
  cmap = sns.color_palette("mako", as_cmap=True, n_colors = 5)
  ax = sns.heatmap(df, cmap=cmap, annot=True, fmt=".2%")
  cbar= ax.collections[0].colorbar
  tick_labels = cbar.ax.get_yticklabels()
  tick_values = cbar.get_ticks()
  for i, tick_label in enumerate(tick_labels):
      tick_label.set_text(f"{int(tick_values[i] * 100)}%")
  cbar.ax.set_yticklabels(tick_labels)

  title = "Matriz de Confusión"
  ax.set_title(title, fontsize=15, pad=10)
  plt.tight_layout()
  plt.savefig("heatmap.png")


def confusion_row_to_p(row):
    total = row.sum()
    return row.apply(lambda x: (x / total).round(4))

def main():
    
    k = 5
    test_p = 0.2
    weighted = False
    remove_na = True
    normalize = True

    df = pd.read_csv("./src/reviews_sentiment.csv", sep=';')

    # (a)
    get_avg_wordcount(df, 1)

    # (b, c)
    knn_classification(df, k, test_p, weighted, remove_na, normalize)

    # METRICS
    partitions = 5

    confusion_matrix = {real: {pred: 0 for pred in range(1, 6)} for real in range(1, 6)}
    precision_per_p = [0 for _ in range(partitions)]

    for p in range(partitions):
        df = pd.read_csv("classification" + str(p) + ".csv")
        curr_correct = calculate_matrix(df, confusion_matrix)
        precision_per_p[p] = curr_correct/len(df)

    confusion_df = pd.DataFrame(confusion_matrix)
    confusion_df = confusion_df.apply(confusion_row_to_p, axis=1)

    #Heatmap
    get_matrix_heatmap(confusion_df)

    mean_std_precision = {"mean": np.mean(precision_per_p), "std": np.std(precision_per_p), "max_precision": max(precision_per_p)}
    print(mean_std_precision)
    pd.DataFrame([mean_std_precision]).to_csv("mean_metrics.csv")

if __name__ == "__main__":
    main()


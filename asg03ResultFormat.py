import pandas as pd


# Transform numerical class names to strings
def result_format(result_file, twid):
    df_sentiment = pd.read_csv(result_file)
    map_dict = {0: 'neg', 1: 'neu', 2: 'pos'}
    df_sentiment['0'] = df_sentiment['0'].map(map_dict)
    df_tweet_id = pd.read_csv(twid)
    df_sentiment.insert(0, column='tweet_id', value=df_tweet_id['tweet_id'])
    df_sentiment.to_csv(result_file, index=False, header=['tweet_id', 'sentiment'])


if __name__ == "__main__":
    result_format('pred_results/lr_predictions.csv', 'data/test_tfidf.csv')

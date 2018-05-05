import pandas as pd
from sklearn import preprocessing
from randomForest import *


def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == h:
            correct += 1.0
    return correct / count


def encode_labels(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['workclass'].unique())
    df['workclass'] = le.transform(df['workclass'])
    df.head()


    le.fit(df['education'].unique())
    df['education'] = le.transform(df['education'])

    le.fit(df['marital-status'].unique())
    df['marital-status'] = le.transform(df['marital-status'])

    le.fit(df['occupation'].unique())
    df['occupation'] = le.transform(df['occupation'])


    le.fit(df['relationship'].unique())
    df['relationship'] = le.transform(df['relationship'])


    le.fit(df['race'].unique())
    df['race'] = le.transform(df['race'])

    le.fit(df['sex'].unique())
    df['sex'] = le.transform(df['sex'])

    le.fit(df['native-country'].unique())
    df['native-country'] = le.transform(df['native-country'])

    le.fit(df['income'].unique())
    df['income'] = le.transform(df['income'])

    return df


df = pd.read_csv('data/adult.data', header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num'
                                                                   , 'marital-status', 'occupation', 'relationship',
                                                                   'race', 'sex', 'capital-gain' , 'capital-loss',
                                                                   'hours-per-week', 'native-country', 'income'])
# Remove nas
df = df.replace(' ?', np.NaN)
df = df.dropna()
print("Length of dataset=", len(df))
df = encode_labels(df)
df_test = pd.read_csv('data/adult_test.txt', sep=",", header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num'
                                                                   , 'marital-status', 'occupation', 'relationship',
                                                                   'race', 'sex', 'capital-gain' , 'capital-loss',
                                                                   'hours-per-week', 'native-country', 'income'])
df_test = df_test.replace(' ?', np.NaN)
df_test = df_test.dropna()
df_test = encode_labels(df_test)

df['binned_age'] = pd.cut(df['age'], bins=3, labels=[0, 1, 2])
series_age, bins_of_age = pd.cut(df['age'], bins=3, retbins=True, labels=False)
df_test['binned_age'] = pd.cut(df_test['age'], bins=bins_of_age, labels=[0, 1, 2])
df['binned_fnlwgt'] = pd.cut(df['fnlwgt'], bins=3, labels=[0, 1, 2])
series_fnlwgt, bins_of_fnlwgt = pd.cut(df['fnlwgt'], bins=3, retbins=True, labels=False)
df_test['binned_fnlwgt'] = pd.cut(df_test['fnlwgt'], bins=bins_of_fnlwgt, labels=[0, 1, 2])
df['binned_capital-gain'] = pd.cut(df['capital-gain'], bins=3, labels=[0, 1, 2])
series_gain, bins_of_cp = pd.cut(df['capital-gain'], bins=3, retbins=True, labels=False)
df_test['binned_capital-gain'] = pd.cut(df_test['capital-gain'], bins=bins_of_cp, labels=[0, 1, 2])
df['binned_capital-loss'] = pd.cut(df['capital-loss'], bins=3, labels=[0, 1, 2])
series_loss, bins_of_loss = pd.cut(df['capital-loss'], bins=3, retbins=True, labels=False)
df_test['binned_capital-loss'] = pd.cut(df_test['capital-loss'], bins=bins_of_loss, labels=[0, 1, 2])
df['binned_hours-per-week'] = pd.cut(df['hours-per-week'], bins=3, labels=[0, 1, 2])
series_hp, bins_of_hp = pd.cut(df['hours-per-week'], bins=3, retbins=True, labels=False)
df_test['binned_hours-per-week'] = pd.cut(df_test['hours-per-week'], bins=bins_of_hp, labels=[0, 1, 2])

# print("Train data after processing=", df.head())
# print("Test data after processing=", df_test.head())

X_train = df[['workclass', 'education', 'education-num', 'marital-status'
       , 'occupation', 'relationship', 'race', 'sex', 'native-country',
      'binned_age', 'binned_fnlwgt', 'binned_capital-gain', 'binned_capital-loss', 'binned_hours-per-week']]
y_train = df['income']

X_train = X_train.values.tolist()
y_train = y_train.values.tolist()

X_test = df_test[['workclass', 'education', 'education-num', 'marital-status'
       , 'occupation', 'relationship', 'race', 'sex', 'native-country',
      'binned_age', 'binned_fnlwgt', 'binned_capital-gain', 'binned_capital-loss', 'binned_hours-per-week']]

X_test = X_test.values.tolist()
y_test = df_test['income']
y_test = y_test.values.tolist()

rf_classifier = random_forest()
rf_classifier.fit(X_train, y_train)
hyp = rf_classifier.predict(X_test)
print("Hypothesis=", hyp)
score = accuracy(y_test, hyp)
print("Accuracy score=", score)
import pandas as pd

train_df = pd.read_csv('./train_preprocessed.csv')
correlation_Sur_sex = train_df['Survived'].corr(train_df['Sex'])
print("The correlation of Survived and Sex: ", correlation_Sur_sex)

sex_mapping = {1: 'Female', 0: 'Male'}
train_df['Sex'] = train_df['Sex'].map(sex_mapping)

print(train_df[['Sex', 'Survived']].groupby(['Sex'],
as_index=False).mean().sort_values(by='Survived', ascending=False))



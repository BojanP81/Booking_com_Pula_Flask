import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Booking_Pula_accommodation_final.csv')
df.drop('Review_Score_', axis=1, inplace=True)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

lin_reg = LinearRegression()
lr = lin_reg.fit(X_train, y_train)
pickle.dump(lr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))



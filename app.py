from flask import Flask, render_template, request
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])


def getvalue():

    import pandas as pd

    from sklearn.model_selection import train_test_split
    data = pd.read_csv(r"C:/Users/singh/Desktop/machine learning cdac/diabetes.csv")
    data = pd.DataFrame(data)
    data = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose != 0)]
    print(data.shape)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome', axis=1), data['Outcome'], test_size=0.30,random_state=101)
    from sklearn.linear_model import LogisticRegression

    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score:', accuracy_score(y_test, predictions))
    '''using k fold'''
    from sklearn.model_selection import KFold

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    kf = KFold(n_splits=3)  # Define the split - into 3 folds
    kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validator
    print(kf)
    lg1 = LogisticRegression()
    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lg1.fit(X_train, y_train)
    pred1 = lg1.predict(X_test)
    from sklearn.metrics import accuracy_score
    print('accuracy score(K-fold):', accuracy_score(y_test, pred1))
    name = request.form['name']

    pregnancy= int(request.form['preg'])
    glucose = int(request.form['glucose'])
    blood = int(request.form['bloodpressure'])
    skin = int(request.form['skin'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    pedigree = float(request.form['pedigree'])
    age = int(request.form['age'])

    new_input = [[pregnancy, glucose, blood, skin, insulin,bmi, pedigree,age]]
    # get prediction for new input
    new_output = lg1.predict(new_input)
    # summarize input and output
    print(new_input, new_output)
    if new_output[0]==1:
        nn=name+', your test result is : positive'
    else:
        nn = name+'your test result is: negative'
    return render_template('pass.html', n=nn)




if __name__ == '__main__':
    app.run(debug=True)
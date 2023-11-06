from flask import Flask , render_template , request
import pickle
import numpy as np

app = Flask(__name__)
model1 = pickle.load(open('ar_svm.pkl','rb'))
ss1=pickle.load(open('ar_ss.pkl','rb'))
le1=pickle.load(open('le1.pkl','rb'))
le2=pickle.load(open('le2.pkl','rb'))
le3=pickle.load(open('le3.pkl','rb'))
le4=pickle.load(open('le4.pkl','rb'))
le5=pickle.load(open('le5.pkl','rb'))
le6=pickle.load(open('le6.pkl','rb'))
le7=pickle.load(open('le7.pkl','rb'))
le8=pickle.load(open('le8.pkl','rb'))
le9=pickle.load(open('le9.pkl','rb'))
le10=pickle.load(open('le10.pkl','rb'))


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/about")
def aboutPage():
    return render_template('about.html')

@app.route("/output",methods=['POST'])
def predicted():
    print("Hello")
    Airline_name=request.form['Airline name']
    print("Hello1",Airline_name)
    Seat_Type=request.form['Seat Type']
    print("Hello2")
    Type_of_Traveller=request.form['Type Of Traveller']
    print("Hello3")
    Origin = request.form['Origin']
    print("Hello4")
    Destination = request.form['Destination']
    print("Hello5")
    Month_Flown = request.form['Month Flown']
    print("Hello6")
    Year_Flown = request.form['Year Flown']
    print("Hello7")
    Verified = request.form['Verified']
    print("Hello8")
    S_C = request.form['S_C']
    print("Hello9")
    F_B=request.form['F_B']
    print("Hello10")
    G_S=request.form['G_S']
    print("Hello11")
    O_R=request.form['O_R']
    print("Hello12")

    data = [[Airline_name,Seat_Type,Type_of_Traveller,Origin,Destination,Month_Flown,Year_Flown,Verified,S_C,F_B,G_S,O_R]]
    Airline_name = le1.fit_transform([Airline_name])[0]
    Seat_Type = le2.fit_transform([Seat_Type])[0]
    Type_of_Traveller = le3.fit_transform([Type_of_Traveller])[0]
    Origin = le4.fit_transform([Origin])[0]
    Destination = le5.fit_transform([Destination])[0]
    Month_Flown = le6.fit_transform([Month_Flown])[0]
    Year_Flown = le7.fit_transform([Year_Flown])[0]
    Verified = le8.fit_transform([Verified])[0]
    O_R = le9.fit_transform([O_R])[0]
    encoded_data = [
        Airline_name,
        Seat_Type,
        Type_of_Traveller,
        Origin,
        Destination,
        Month_Flown,
        Year_Flown,
        Verified,
        S_C,F_B,G_S,
        O_R,
    ]
    prediction=model1.predict(ss1.transform([encoded_data]))
    if prediction == 1 :
        a="Recommended"
        return render_template('index.html',y=a)
    else :
        b="Not Recommended"
        return render_template('index.html',y=b)
    

if __name__=="__main__":
    app.debug = True
    app.run(host="0.0.0.0")

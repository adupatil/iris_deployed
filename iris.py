from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import NumberRange
import numpy as np  
from tensorflow.keras.models import load_model
import joblib
app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'


def return_prediction(model,scaler,sample_json):
	s_len = sample_json['sepal_length']
	s_wid = sample_json['sepal_width']
	p_len = sample_json['petal_length']
	p_wid = sample_json['petal_width']

	flower = [[s_len,s_wid,p_len,p_wid]]

	flower = scaler.transform(flower)

	prediction = model.predict_classes(flower)

	classes = np.array(['setosa', 'versicolor', 'virginica'])

	return classes[prediction[0]]

class FlowerForm(FlaskForm):

	sep_len =  StringField("sepal_length")
	sep_wid =  StringField("sepal_width")
	pet_len =  StringField("petal_length")
	pet_wid =  StringField("petal_width")

	submit = SubmitField("Ananlyze")

model= load_model("final_iris_model.h5")
scaler = joblib.load("iris_scaler.pkl")

@app.route('/',methods=['GET',"POST"])

def index():
	form = FlowerForm()

	if form.validate_on_submit():
		session['sep_len'] = form.sep_len.data
		session['sep_wid'] = form.sep_wid.data
		session['pet_len'] = form.pet_len.data
		session['pet_wid'] = form.pet_wid.data

		return redirect(url_for("predict"))
	return render_template('home.html', form=form)

@app.route('/prediction')
def predict():
	content = {}
	content['sepal_length'] = float(session['sep_len'])
	content['sepal_width'] = float(session['sep_wid'])
	content['petal_length'] = float(session['pet_len'])
	content['petal_width'] = float(session['pet_wid'])

	results = return_prediction(model,scaler,content)

	return render_template("prediction.html",results=results)


if __name__=='__main__':
	app.run()

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import pandas as pd



app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: 'Normal',
1: 'Pancreatic Tumor',
 

}


model = load_model('save.h5')

random = pickle.load(open('pancreatic_random.pkl','rb'))

naive = pickle.load(open('pancreatic_naive.pkl','rb'))

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name [classes_x[0]]

 

@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)
		#plt.imshow(img)
		predict_result = predict_label(img_path)
		 

		#print(predict_result)
	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/chart")
def chart():
	return render_template('chart.html') 

@app.route("/performance")
def performance():
	return render_template('performance.html')  	

@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index(pd.RangeIndex(start=0, stop=len(df)), inplace=True)
        return render_template("preview.html",df_view = df)	


@app.route('/home')
def home():
   return render_template('home.html')

@app.route('/predictions', methods = ['GET', 'POST'])
def predictions():
    return render_template('predictions.html')


@app.route('/upload')
def upload_file():
  return render_template('BatchPredict.html')



@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        Patient_Cohort = request.form['Patient_Cohort']
        Sample_Origin = request.form['Sample_Origin']
        Age = request.form['Age']
        Sex = request.form['Sex']
        Stage = request.form['Stage'] 
        Benign_Sample_Diagnosis = request.form['Benign_Sample_Diagnosis']
        Plasma_CA19_9 = request.form['Plasma_CA19_9']
        Creatinine = request.form['Creatinine']
        Lyu = request.form['LYVE1']
        Reb = request.form['REG1B']
        tt = request.form['TFF1']
        rea = request.form['REG1A']
        
        
        model = request.form['Model']
        
		# Clean the data by convert from unicode to float 
        
        sample_data = [Patient_Cohort,Sample_Origin,Age,Sex,Stage,Benign_Sample_Diagnosis,Plasma_CA19_9,Creatinine,Lyu,Reb,tt,rea]
        print(sample_data)
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
        int_feature = [float(i) for i in sample_data]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model == 'RandomForestClassifier':
           result_prediction = random.predict(ex1)
           
            
        elif model == 'naivebayes':
          result_prediction = naive.predict(ex1)
           
           
        
        if result_prediction == 1:
            result = 'Control  (no pancreatic disease)'
        elif result_prediction == 2:
            result = 'Benign (benign hepatobiliary disease)'  
        elif result_prediction == 3:
            result = 'PDAC (Pancreatic ductal adenocarcinoma)'  
          

    return render_template('predictions.html', prediction_text= result, model = model)
@app.route('/performances')
def performances():
	return render_template('performances.html')   
	

	
if __name__ =='__main__':
	app.run(debug = True)


	

	



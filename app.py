from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

REQUIRED_FIELDS=['gender','ethnicity','parental_level_of_education','lunch','test_preparation_course',
                 'reading_score','writing_score']

def validate_form(form):
    '''
    Returns an error message, or None if the form is valid
    '''
    missing=[field for field in REQUIRED_FIELDS if not form.get(field)]
    if missing:
        return f"Missing fields: {', '.join(missing)}"

    for field in ['reading_score','writing_score']:
        try:
            score=float(form.get(field))
        except ValueError:
            return f"{field} must be a number"
        if not 0<=score<=100:
            return f"{field} must be between 0 and 100"

    return None

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        error=validate_form(request.form)
        if error:
            return render_template('home.html',error=error),400

        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=round(float(results[0]),2))


if __name__=="__main__":
    print("App running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0")

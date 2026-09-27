import hmac
import os
import time

from flask import Flask,request,render_template

from src import monitoring
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Prometheus: request metrics + GET /metrics
monitoring.init_app(app)

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

## Health check for Docker / cloud platforms

@app.route('/health')
def health():
    # model is None until the first prediction (the model is loaded lazily)
    return {"status":"ok","model":PredictPipeline.model_info()}

## Reload the champion model without restarting the app (after a new promotion in the registry)

@app.route('/admin/reload-model',methods=['POST'])
def reload_model():
    expected_token=os.getenv("RELOAD_TOKEN")
    if not expected_token:
        # disabled unless a token is configured on the server
        return {"error":"not found"},404

    auth=request.headers.get("Authorization","")
    if not hmac.compare_digest(auth,f"Bearer {expected_token}"):
        return {"error":"unauthorized"},401

    return {"status":"reloaded","model":PredictPipeline.reload_model()}

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
            monitoring.observe_invalid(error)
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
        start=time.perf_counter()
        results=predict_pipeline.predict(pred_df)
        monitoring.observe_prediction(pred_df,float(results[0]),time.perf_counter()-start)
        return render_template('home.html',results=round(float(results[0]),2))


if __name__=="__main__":
    print("App running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0")

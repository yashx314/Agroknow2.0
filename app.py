# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup ,redirect
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch 
from torchvision import transforms
from PIL import Image  
from utils.model import ResNet9


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

prediction_image = {
"rice":"https://media0.giphy.com/media/iqkUhBIdNOetJWXtNx/giphy.gif",
"maize":"https://media.tenor.com/lCZqZJwGOwUAAAAM/misscanon-misscornon.gif",
"chickpea":"https://myheartbeets.com/wp-content/uploads/2018/01/chana-masala-GIF_10.gif",
"kidneybeans":"https://moonandspoonandyum.com/wp-content/uploads/2021/08/ezgif.com-gif-maker-30.gif",
"pigeonpeas":"https://64.media.tumblr.com/d0fd1b116bada7e122e7e571f1ef49ca/255e0be896f98749-ae/s400x600/be59056211f2f16825f681d1f6f32df753536289.gif",
"mothbeans":"https://j.gifs.com/4QyNWx.gif",  
"mungbean":"https://j.gifs.com/6XAP0O@facebook.gif",
"Blackgram":"https://1.bp.blogspot.com/-HbQP7FGDlqs/X246bse0ILI/AAAAAAAAJYY/6a0URspK-sU2tOIz5AJ-iovfNiHV_51gQCLcBGAsYHQ/s368/InShot_20200925_043745277_1.gif",
"lentil":"https://media1.giphy.com/media/ZaVikY6sxRiNtBxsjV/giphy.gif",
"pomegranate":"https://64.media.tumblr.com/f7714fd4abe7a3f80b0ac452b28118b9/33d40c54e8d557ed-fe/s400x600/dcf9c7d54dba1f90cb71a6deaa87dde7c88c8f22.gif",
"banana":"https://usagif.com/wp-content/uploads/gifs/banana-10.gif",
"mango":"https://media.tenor.com/42NXguB_OPcAAAAd/mango-omg-mango.gif",
"grapes":"https://i.makeagif.com/media/9-03-2015/CsJmGb.gif",
"watermelon":"https://j.gifs.com/mQx5Z9.gif",
"muskmelon":"https://media.tenor.com/ZXHVXn2dVNMAAAAC/melon-weed-em-and-reap.gif",
"apple":"https://i.pinimg.com/originals/5c/81/51/5c8151f5751ded2772fa8ba4594dc5ba.gif",
"orange":"https://media1.giphy.com/media/3gL1axFVLCmxGEiLoz/giphy.gif",
"papaya":"https://media.tenor.com/vw6qQhs1FdsAAAAC/papaya.gif",
"coconut":"https://assets.website-files.com/602e36fc951f62605d5e305a/6151c71d37d7f91b3aa43622_coconut.gif",
"cotton":"https://i.gifer.com/NaDO.gif",
"jute":"https://www.localguidesconnect.com/t5/image/serverpage/image-id/1008207i0902C53201888F88?v=v2",
"coffee":"https://media.tenor.com/ejOgvH2pGEsAAAAC/beans-coffee.gif",  

}

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Agroknow - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Agroknow - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Agroknow - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)
  
# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        print(request.form)
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            # print(prediction_image[final_prediction])
            final_prediction_image = prediction_image[final_prediction]
            return render_template('crop-result.html', prediction=final_prediction, title=title,crop_image_link=final_prediction_image)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"


    # response="apple"
    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agroknow - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')  
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@app.route("/redirect-external", methods=["GET"])
def dashboard():
    return redirect("http://localhost:8501/", code=302)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)

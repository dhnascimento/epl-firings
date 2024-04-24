import numpy as np
from flask import Flask, request, render_template
from newspaper import Article
import dill

app = Flask(__name__)
# Load the model
model = dill.load(open('model.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    animation = 1
    if request.method == 'GET':
        animation = 1
        return render_template('index.html', animation = animation)
    else:
        animation = 0
        # Get the data from the POST request.
        url = request.form['url']
        # Parse download and parse article text
        article = Article(url)
        article.download()
        article.parse()

        text = article.text
        title = article.title
        # Call model to run prediction
        prediction = model.predict([text])
        prediction2 = str(100 * round(model.predict_proba([text])[0][prediction[0]],4))

        # Take the first value of prediction
        output = "BE FIRED" if prediction[0] == 1 else "NOT BE FIRED"
        return render_template(
            'index.html',
            prediction_text = output,
            prob = prediction2,
            title = title,
            url = url,
            text = text,
            animation = animation
        )

if __name__ == '__main__':
    app.run(port=5000, debug=True)
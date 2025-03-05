from flask import Flask, request, render_template, g, jsonify
# Task 1 --- Import Libraries


# Task 12: Import the functions from utility.py


# Class mappings
class_mapping = {'Downdog': 0, 'Goddess': 1, 'Plank': 2, 'Tree': 3, 'Warrior': 4}
app = Flask(__name__)

# Task 14 --- Configure the upload folder


@app.route('/')
def index():
    return render_template('index.html')

# Task 15 --- Define route to process the request for processing endoint


def process(): 
    # Task 14 --- Define the function to process image from front end
    

    # Task 16 --- Display Pose Detection Results


    return

# Task 12 --- Predict class of a given image


if __name__ == '__main__':
    # Task 12: Use the Model


    app.run(debug=True)
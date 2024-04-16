from flask import Flask, render_template, request, jsonify
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from PIL import Image
import torch

app = Flask(__name__)

# Load pre-trained LXMERT model and tokenizer
# tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
# model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")
# model.load_state_dict(torch.load("snap/pretrained/model_LXRT.pth", map_location=torch.device('cpu')))
# model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'question' not in request.form:
        return 'Error: Please provide both an image and a question.'

    image = request.files['image']
    question = request.form['question']

    # Process image
    image = Image.open(image)
    # Resize or preprocess image as needed

    # Tokenize inputs
    inputs = tokenizer(image, question, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Decode the predicted answer
    predicted_answer_idx = outputs['question_answering_output'][0].argmax().item()
    predicted_answer = tokenizer.decode(predicted_answer_idx)

    return jsonify({'answer': predicted_answer})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import torch

app = Flask(__name__)
model = torch.load("tree_counting_model.pth", map_location=torch.device('cpu'))

def count_trees(image):
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(image)

    # Count trees (example code, replace with your own logic)
    tree_count = len(output)  # Change this to actual tree counting logic
    return tree_count

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get uploaded image
        image_file = request.files['image']
        if image_file:
            # Load image
            image = Image.open(image_file)

            # Count trees
            tree_count = count_trees(image)

            # Render template with results
            return render_template('index.html', tree_count=tree_count)

    # Render empty form
    return render_template('index.html', tree_count=None)

if __name__ == '__main__':
    app.run(debug=True)

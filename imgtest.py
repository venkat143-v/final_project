import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from PIL import Image
import os 

loaded_model = tf.keras.models.load_model('ec2Types.h5', compile=False)
loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])



def gif_to_jpg(gif_path,output_folder):
    gif = Image.open(gif_path)
    for frame_number in range(gif.n_frames):
        gif.seek(frame_number)
        frame = gif.convert('RGB')
        frame.save(f"{output_folder}\\modified.jpg")

def png_to_jpg(img_path):
    im = Image.open(img_path) 
    rgb_im = im.convert("RGB") 
    rgb_im.save(r"E:\newfinal\static\data\modified.jpg")

image_path = r"E:\newfinal\static\data\Transistor (1).gif"
file_extension = os.path.splitext(image_path)[-1].lower()

if file_extension in [".png"]:
    png_to_jpg(image_path)
    image = Image.open(r"E:\newfinal\static\data\modified.jpg")
    
elif file_extension in [".gif"]:
    gif_to_jpg(image_path, r"E:\newfinal\static\data")
    image = Image.open(r"E:\newfinal\static\data\modified.jpg")
else:
    image = Image.open(image_path)


# Preprocess the image
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
classes=['Capacitor', 'IC', 'Resistor', 'Transistor']
# Make predictions
predictions = loaded_model.predict(img_array)
class_labels = classes
score = tf.nn.softmax(predictions[0])

print(f"{class_labels[tf.argmax(score)]}")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import streamlit as st
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as models
import os
import glob
import pathlib
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

IMG_HEIGHT, IMG_WIDTH = 299, 299
category_type = ["mathura", "gandhara"]
depiction_type = ["buddha", "bodhisattva", "vishnu", "yakshi", "other"]
medium_type = ["red_sandstone", "phyllite", "schist", "other"]

def main():
    st.sidebar.title("Mathura or Gandhara?")
    st.sidebar.markdown("**02.128 DH Project**")
    st.sidebar.markdown("Identifying a sculpture of Mathura or Gandhara style.")
    run_the_app()
    
def parse_preds(preds):
    category = category_type[np.argmax(preds[0])]
    depiction = depiction_type[np.argmax(preds[1])]
    medium = medium_type[np.argmax(preds[2])]
    output = "**" + category + "** sculpture of **" + depiction + "** made of **" + medium + "** ."
    return output.replace("red_sandstone", "red/pink sandstone")

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    train_dataset, valid_dataset, max_cat = load_dataset()
    st.sidebar.subheader("Select an Image")
    st.sidebar.markdown("The model is trained on images in the Training set. Choose the **Validation** set to test images that the model is not trained on.")
    option = st.sidebar.selectbox("Select from either Training and Validation sets",
                                  ("Train", "Validation"))
    if option == "Train":
        dataset = train_dataset
    elif option == "Validation":
        dataset = valid_dataset
    selected_index = st.sidebar.slider("Slide to pick an image from the " + option + " dataset", 0, len(dataset)-1, 0)
    image, labels = dataset[selected_index]
    preds = infer(image, max_cat)
    truth = parse_preds(labels)
    preds = parse_preds(preds)
    if truth == preds:
        caption = "Model **_correctly_** predicted that art is " + preds
    else:
        caption = "Model **_wrongly_** predicted that art is " + preds + "\n\nThe **_correct_** label is: " + truth
    image_numpy = np.clip(image+0.5, 0.0, 1.0)
    st.image(image_numpy, width=500)
    st.markdown(caption)


    
@st.cache(allow_output_mutation=True)
def load_dataset():
    BATCH_SIZE = 8
    image_files = []
    for file in glob.glob("./dataset/*"):
        image_files.append(file)
    num_images = len(image_files)
    labels_id = {}
    num_type = len(category_type)
    num_dep = len(depiction_type)
    num_med = len(medium_type)
    max_cat = max([num_type, num_dep, num_med])
    with open("./labels.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                labels_id[row[0]] = [row[1], row[2], row[3]]
    def get_label(file_path):
        img_id = file_path.split("/")[-1].replace(".jpg", "").replace(".JPG", "")
        _type, _depict, _medium = labels_id[img_id]
        if _medium not in medium_type:
            _medium = "other"
        if _depict not in depiction_type:
            _depict = "other"
        label = [category_type.index(_type), depiction_type.index(_depict), medium_type.index(_medium)]
        return label
    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    label_list = []
    for file_path in image_files:
        label = get_label(file_path)
        label_list.append(label)
    def process_example_val(img_path, labels):
        img = tf.io.read_file(img_path)
        img = decode_img(img)
        img = tf.clip_by_value(img, 0.0, 1.0) - 0.5
        type_label = tf.one_hot(labels[0], max_cat)
        med_label = tf.one_hot(labels[1], max_cat)
        dep_label = tf.one_hot(labels[2], max_cat)
        return img, (type_label, med_label, dep_label)
    num_total = len(image_files)
    splits = pickle.load( open( "splits.pickle", "rb" ) )
    train_image_files = splits["train_image_files"]
    valid_image_files = splits["valid_image_files"]
    train_label_list = splits["train_label_list"]
    valid_label_list = splits["valid_label_list"]
    num_train = len(train_image_files)
    num_valid = len(valid_image_files)
    train_dataset = []
    valid_dataset = []
    for img_path, labels in zip(train_image_files, train_label_list):
        img, labels = process_example_val(img_path, labels)
        train_dataset.append([img, labels])
    for img_path, labels in zip(valid_image_files, valid_label_list):
        img, labels = process_example_val(img_path, labels)
        valid_dataset.append([img, labels])
    return train_dataset, valid_dataset, max_cat
    

def infer(image, max_cat):
    @st.cache(allow_output_mutation=True)
    def load_pretrained_network():
        input_layer = layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3,), name="input_layer")
        base = models.MobileNetV2(input_tensor=input_layer, include_top=False, weights='imagenet')
        base.trainable = True
        base_output = base.output
        vis_map = layers.Conv2D(filters=3, kernel_size=(1,1))(base_output)
        vis_up = layers.Conv2D(filters=1024, kernel_size=(1,1))(vis_map)
        x = layers.GlobalAveragePooling2D()(vis_up)
        x = layers.Dropout(0.2)(x)
        type_pred = layers.Dense(max_cat, name="type_pred", activation="softmax")(x)
        dep_pred = layers.Dense(max_cat, name="dep_pred", activation="softmax")(x)
        med_pred = layers.Dense(max_cat, name="med_pred", activation="softmax")(x)
        model = tf.keras.models.Model(inputs=input_layer,
                                      outputs=[type_pred, dep_pred, med_pred])
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, decay=0.001)
        model.load_weights("./best_weights.h5")
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      optimizer=opt,
                      metrics=["accuracy"])
        @tf.function(experimental_compile=True)
        def forward(image):
            preds = model(tf.reshape(image, [1,IMG_WIDTH,IMG_HEIGHT,3]))
            return preds
        return model, forward
    model, forward = load_pretrained_network()
    preds = forward(image)
    return preds

if __name__ == "__main__":
    main()
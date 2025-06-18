# style_transfer.py
import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_process_image(image, max_dim=512):
    image = image.convert("RGB")
    long = max(image.size)
    scale = max_dim / long
    image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.LANCZOS)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.keras.applications.vgg19.preprocess_input(image * 255.0)
    return tf.convert_to_tensor(image)[tf.newaxis, :]

def deprocess_image(img):
    x = img[0].numpy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layers = ['block5_conv2']
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return tf.keras.Model([vgg.input], outputs), style_layers, content_layers

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def get_features(model, content_image, style_image, style_layers, content_layers):
    outputs_content = model(content_image)
    outputs_style = model(style_image)
    style_features = [gram_matrix(outputs_style[i]) for i in range(len(style_layers))]
    content_features = [outputs_content[i + len(style_layers)] for i in range(len(content_layers))]
    return style_features, content_features

def compute_loss(outputs, style_features, content_features, style_weight=1e-2, content_weight=1e4):
    style_outputs = outputs[:len(style_features)]
    content_outputs = outputs[len(style_features):]
    style_loss = tf.add_n([
        tf.reduce_mean((style_outputs[i] - style_features[i]) ** 2)
        for i in range(len(style_features))
    ])
    content_loss = tf.add_n([
        tf.reduce_mean((content_outputs[i] - content_features[i]) ** 2)
        for i in range(len(content_features))
    ])
    return style_weight * style_loss + content_weight * content_loss

@tf.function
def train_step(image, extractor, style_features, content_features, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = compute_loss(outputs, style_features, content_features)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, -103.939, 255.0 - 103.939))

def run_style_transfer(content_image, style_image, epochs=250):
    model, style_layers, content_layers = get_model()
    content_tensor = load_and_process_image(content_image)
    style_tensor = load_and_process_image(style_image)
    generated_image = tf.Variable(content_tensor)

    style_features, content_features = get_features(
        model, content_tensor, style_tensor, style_layers, content_layers
    )
    optimizer = tf.optimizers.Adam(learning_rate=0.02)

    for _ in range(epochs):
        train_step(generated_image, model, style_features, content_features, optimizer)

    return deprocess_image(generated_image)
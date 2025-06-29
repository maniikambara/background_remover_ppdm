import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
import numpy as np

def residual_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    s = Conv2D(num_filters, 1, padding="same")(inputs)
    s = BatchNormalization()(s)
    x = Activation("relu")(x + s)
    return x

def dilated_conv_block(inputs, num_filters):
    x1 = Conv2D(num_filters, 3, padding="same", dilation_rate=3)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    
    x2 = Conv2D(num_filters, 3, padding="same", dilation_rate=6)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    
    x3 = Conv2D(num_filters, 3, padding="same", dilation_rate=9)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    
    x = Concatenate()([x1, x2, x3])
    x = Conv2D(num_filters, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters)
    return x

def build_unet_model(input_shape=(512, 512, 3)):
    inputs = Input(input_shape)
    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Get correct layer names
    s1 = inputs
    s2 = backbone.get_layer("conv1_relu").output
    s3 = backbone.get_layer("conv2_block3_out").output
    s4 = backbone.get_layer("conv3_block4_out").output
    s5 = backbone.get_layer("conv4_block6_out").output
    
    # Bridge
    bridge = dilated_conv_block(s5, 1024)
    
    # Decoder
    d1 = decoder_block(bridge, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Multi-scale outputs
    y1 = UpSampling2D((8, 8), interpolation="bilinear")(d1)
    y1 = Conv2D(1, 1, padding="same", activation="sigmoid")(y1)
    
    y2 = UpSampling2D((4, 4), interpolation="bilinear")(d2)
    y2 = Conv2D(1, 1, padding="same", activation="sigmoid")(y2)
    
    y3 = UpSampling2D((2, 2), interpolation="bilinear")(d3)
    y3 = Conv2D(1, 1, padding="same", activation="sigmoid")(y3)
    
    y4 = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    
    outputs = Concatenate()([y1, y2, y3, y4])
    model = Model(inputs, outputs, name="ResNet50-UNet")
    return model

def test_model():
    try:
        print("Creating model...")
        model = build_unet_model((512, 512, 3))
        
        print("Model created successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        print("Testing with dummy input...")
        dummy_input = np.random.random((1, 512, 512, 3)).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        print(f"Output shape: {output.shape}")
        print("Model test passed")
        
        return model
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    model = test_model()
    
    if model is not None:
        print("\nMODEL SUMMARY")
        print("=" * 50)
        model.summary()
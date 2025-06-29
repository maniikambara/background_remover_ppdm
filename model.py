# type: ignore
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def conv_bn_act(x, filters, kernel_size=3, dilation_rate=1, activation='relu'):
    """Convolution + BatchNorm + Activation block"""
    x = Conv2D(filters, kernel_size, padding="same", dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    return Activation(activation)(x)

def residual_block(inputs, filters):
    """Residual block with skip connection"""
    x = conv_bn_act(inputs, filters)
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    skip = conv_bn_act(inputs, filters, kernel_size=1, activation=None)
    return Activation("relu")(x + skip)

def dilated_conv(inputs, filters):
    """Multi-scale dilated convolution block"""
    branches = [conv_bn_act(inputs, filters, dilation_rate=rate) for rate in [3, 6, 9]]
    x = Concatenate()(branches)
    return conv_bn_act(x, filters, kernel_size=1)

def decoder_block(inputs, skip_features, filters):
    """Decoder block with upsampling and skip connection"""
    x = UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip_features])
    return residual_block(x, filters)

def build_model(input_shape):
    """Build U-Net model with ResNet50 encoder"""
    inputs = Input(input_shape)
    
    # Encoder (ResNet50)
    resnet = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    skip_layers = ["input_layer", "conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    skips = [resnet.get_layer(name).output for name in skip_layers]
    
    # Bridge
    bridge = dilated_conv(skips[-1], 1024)
    
    # Decoder
    filters = [512, 256, 128, 64]
    x = bridge
    decoder_outputs = []
    
    for i, (skip, f) in enumerate(zip(reversed(skips[1:-1]), filters)):
        x = decoder_block(x, skip, f)
        decoder_outputs.append(x)
    
    # Final decoder block
    x = decoder_block(x, skips[0], filters[-1])
    decoder_outputs.append(x)
    
    # Multi-scale outputs
    upsampling_factors = [8, 4, 2, 1]
    outputs = []
    
    for decoder_out, factor in zip(decoder_outputs, upsampling_factors):
        if factor > 1:
            y = UpSampling2D((factor, factor), interpolation="bilinear")(decoder_out)
        else:
            y = decoder_out
        y = Conv2D(1, 1, padding="same", activation="sigmoid")(y)
        outputs.append(y)
    
    final_output = Concatenate()(outputs)
    return Model(inputs, final_output, name="U-Net")

if __name__ == "__main__":
    model = build_model((512, 512, 3))
    model.summary()
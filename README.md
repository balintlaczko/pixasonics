# Pixasonics: An Image Sonification Toolbox for Python

![Status](https://img.shields.io/pypi/status/pixasonics) [![Version](https://img.shields.io/pypi/v/pixasonics)](https://pypi.org/project/pixasonics/) [![Binder](https://2i2c.mybinder.org/badge_logo.svg)](https://2i2c.mybinder.org/v2/gh/balintlaczko/pixasonics/main?urlpath=%2Fdoc%2Ftree%2Fpixasonics_proto.ipynb)

Some test images are included from the [CELLULAR dataset](https://zenodo.org/records/8315423).

# Introduction

Pixasonics is a library for interactive audiovisual image analysis and exploration, through image sonification. That is, it is using real-time audio and visualization to listen to image data: to map between image features and acoustic parameters. This can be handy when you need to work with a large number of images, image stacks, or hyper-spectral images (involving many color channels) where visualization becomes limiting, challenging, and potentially overwhelming.

With pixasonics, you can launch a little web application (running in a Jupyter notebook), where you can load images, probe their data with various feature extraction methods, and map the extracted features to parameters of synths, devices that make sound. You can do all this in real-time, using a visual interface, you can remote-control the interface programmatically, record sound real-time, or non-real-time, with a custom script.

# Installation

```
pip install pixasonics
```

# Quick launch

After you installed pixasonics, you can launch the tutorial Jupyter notebook from the Terminal:
```
pixasonics-notebook
```
This will launch a local version of [this tutorial notebook](https://github.com/balintlaczko/pixasonics/blob/main/pixasonics/pixasonics_tutorial.ipynb).

# If you are in a hurry...

```python
from pixasonics.core import App, Mapper
from pixasonics.features import MeanChannelValue
from pixasonics.synths import Theremin

# create a new app
app = App()

# load an image from file
app.load_image_file("images/test.jpg")

# create a Feature that will report the mean value of the red channel
mean_red = MeanChannelValue(filter_channels=0, name="MeanRed")
# attach the feature to the app
app.attach(mean_red)

# create a Theremin synth
theremin = Theremin(name="MySine")
# attach the Theremin to the app
app.attach(theremin)

# create a Mapper that will map the mean red pixel value to Theremin frequency
red2freq = Mapper(mean_red, theremin["frequency"], exponent=2, name="Red2Freq")
# attach the Mapper to the app
app.attach(red2freq)
```

# Toolbox Structure

Pixasonics (at the moment) is expected to run in a Jupyter notebook environment. (Nothing stops you from using it in the terminal, but it is not optimized for that yet.)

At the center of pixasonics is the `App` class. This represents a template pipeline where all your image data, feature extractors, synths and mappers will live. The App also comes with a graphical user interface (UI). At the moment it is expected that you only create one `App` at a time, which will control the global real-time audio server. (And every time you create an `App` it will reset the audio graph.)

When you have your app, you load an image (either from a file, or from a numpy array) which will be displayed in the `App` canvas. Note that _currently_ your image data height and width dimensions (the first two) will be downsampled to the `App`'s `image_size` creation argument, which is a tuple of `(500, 500)` pixels by default.

Then you can explore the image data with a Probe (represented by the yellow rectangle on the canvas) using your mouse or trackpad. The Probe is your "stethoscope" on the image, and more technically, it is the sub-matrix of the Probe that is passed to all `Feature` objects in the pipeline.

Speaking of which, you can extract visual features using the `Feature` base class, or any of its convenience abstractions (e.g. `MeanChannelValue`). Currently only basic statistical reductions are supported, such as `"mean"`, `"median"`, `"min"`, `"max"`, `"sum"`, `"std"` (standard deviation) and `"var"` (variance). `Feature` objects also come with a UI that shows their current values and global/running min and max. There can be any number of different `Feature`s attached to the app, and all of them will get the same Probe matrix as input.

Image features are to be mapped to synthesis parameters, that is, to the settings of sound-making gadgets. (This technique is called "Parameter Mapping Sonification" in the literature.) All synths (and audio) in pixasonics are based on the fantastic [signalflow library](https://signalflow.dev/). For now, there are 5 synth classes that you can use (and many more are on the way): `Theremin`, `Oscillator`, `FilteredNoise`, and `SimpleFM`. Each synth comes with a UI, where you can tweak the parameters (or see them being modulated by `Mapper`s) in real-time.

What connects the output of a `Feature` and the input parameter of a Synth is a `Mapper` object. There can be multiple `Mapper`s reading from the same `Feature` buffer and a Synth can have multiple `Mapper`s modulating its different parameters.

# How to contribute

If you encounter any funky behavior, please open an [issue](https://github.com/balintlaczko/pixasonics/issues)!
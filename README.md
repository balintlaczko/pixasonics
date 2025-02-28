# Pixasonics: An Image Sonification Toolbox for Python

![Status](https://img.shields.io/pypi/status/pixasonics) [![Version](https://img.shields.io/pypi/v/pixasonics)](https://pypi.org/project/pixasonics/) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/balintlaczko/pixasonics/HEAD?urlpath=%2Fdoc%2Ftree%2Fpixasonics%2Fpixasonics_tutorial.ipynb)

![A screenshot of the graphical user interface](pixasonics/figures/interface_screenshot.png)

Some test images (such as the one in the above figure) are included from the [CELLULAR dataset](https://zenodo.org/records/8315423).

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

Pixasonics mainly designed to run in a Jupyter notebook environment. (It does also work in command line scripts.)

At the center of pixasonics is the `App` class. This represents a template pipeline where all your image data, feature extractors, synths and mappers will live. The App also comes with a graphical user interface (UI). You can do a lot with a single `App` instance, but nothing stops you from spawning different `App`s with bespoke setups.

When you have your app, you load an image (either from a file, or from a numpy array) which will be displayed in the `App` canvas. Note that _currently_ your image data height and width dimensions (the first two) will be downsampled to the `App`'s `image_size` creation argument, which is a tuple of `(500, 500)` pixels by default. (This will be improved later, stay tuned!)

Then you can explore the image data with a Probe (represented by the yellow rectangle on the canvas) using your mouse or trackpad. The Probe is your "stethoscope" on the image, and more technically, it is the sub-matrix of the Probe that is passed to all `Feature` objects in the pipeline.

Speaking of which, you can extract visual features using the `Feature` base class, or any of its convenience abstractions (e.g. `MeanChannelValue`). All basic statistical reductions are supported, such as `"mean"`, `"median"`, `"min"`, `"max"`, `"sum"`, `"std"` (standard deviation) and `"var"` (variance), but you can also make your own custom feature extractors by inheriting from the `Feature` base class (stay tuned for a K-means clustering example in the Advanced Use Cases section!). `Feature` objects also come with a UI that shows their current values and global/running min and max. There can be any number of different `Feature`s attached to the app, and all of them will get the same Probe matrix as input.

Image features are to be mapped to synthesis parameters, that is, to the settings of sound-making gadgets. (This technique is called "Parameter Mapping Sonification" in the literature.) All `Synth`s (and audio) in pixasonics are based on the fantastic [signalflow library](https://signalflow.dev/). For now, there are 5 `Synth` classes that you can use (and many more are on the way): `Theremin`, `Oscillator`, `FilteredNoise`, and `SimpleFM`. Each `Synth` comes with a UI, where you can tweak the parameters (or see them being modulated by `Mapper`s) in real-time.

What connects the output of a `Feature` and the input parameter of a Synth is a `Mapper` object. There can be multiple `Mapper`s reading from the same `Feature` buffer and a `Synth` can have multiple `Mapper`s modulating its different parameters.

# Advanced Use Cases

There are a few "breakout" doors designed to integrate pixasonics in your existing workflow or to speed/scale up your sonification sessions:
- __Loading Numpy arrays__: This lets you load any matrix data (up to 4 dimensions) as an image to sonify. If you have any specific preprocessing, you can set it up to output Numpy matrices which you can then load into an `App`. Using Numpy arrays also lets you load image sequences or hyper-spectral images (there is no conceptual restriction of the number of color channels or image layers used). By the way, if you don't want to worry about Numpy arrays, you can also directly load HDR images from files.
- __Non-real-time rendering__: Instead of having to move the Probe in real-time, perhaps for a longer recording, you can script a "timeline" and render it non-real-time. You can also reuse a script to render the same scan pattern on many images.
- __Headless mode__: While the `App` class is meant to help with interactive audiovisual exploration, you can totally skip its entire graphical user interface, and remote-control it using its properties (maybe using OSC messages coming from a different process). You should also use headless mode if you are outside of a Jupyter Notebook environment, and using pixasonics in a script.
- __Multichannel `Synth`s__: Providing a list instead of a number for any of a `Synth`s arguments will make it multichannel, which can be used to sonify `Feature`s that have more than one number. And don't worry if the number of features do not match the number of `Synth` channels: in these cases `Mapper`s will dynamically resample the feature vector to fit the number of channels.
- __`Feature` base class and custom `Feature`s__: While there are lots of convenient abstractions for simple `Feature`s (e.g. `MeanChannelValue`, `MedianRowValue`, etc), these are all just configs for the `Feature` base class, and if you learn how it works, you can intuitively fit the `Feature` to whatever slice of the image you need to focus on, using any of the "built-in" (Numpy-based) reducing methods. But you can also create your completely custom `Feature` processors (let's say one that fits a K-means model on the image) by inheriting from the `Feature` base class and overriding two of its methods.
- __Multiple `App`s in the same session__: You can also set up different `App`s with different pipelines (or even images) and use them simultaneously in the same Notebook. For scientists, this can help testing the same sonifications on different images (or sequences), or different sonification setups on the same image data. For creatives, this will let you create different interactive instruments.

# How to contribute

If you encounter any funky behavior, please open an [issue](https://github.com/balintlaczko/pixasonics/issues)!
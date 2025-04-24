import os
from pixasonics.core import App, Mapper
from pixasonics.features import MeanChannelValue
from pixasonics.synths import Theremin

# a for loop where for each image in the folder we load a headless app and render a timeline in nrt mode
img_folder = "pixasonics/images/cellular_dataset/merged_8bit/"
img_files = os.listdir(img_folder)

# example: horizontal scan
duration = 5
my_timeline = [
    (0, {
        "probe_width": 1,
        "probe_height": 500,
        "probe_x": 0,
        "probe_y": 0
    }),
    (duration, {
        "probe_x": 499
    })
]
app = App(headless=True, nrt=True) # create a global graph object, necessary for the Theremin
# only need to create processor objects once and attach them to the apps
mean_red = MeanChannelValue(filter_channels=0, name="MeanRed") # we will attach it later, when there is an image loaded
theremin = Theremin(name="MySine")
red2freq = Mapper(mean_red, theremin["frequency"], exponent=2, name="Red2Freq")
# these we can already attach
app.attach(theremin)
app.attach(red2freq)
# loop over all images in the folder, create a headless app, load the image, attach the processors and render the timeline
for img_file in img_files:
    print(f"Processing {img_file}")
    img_path = os.path.join(img_folder, img_file)
    app.load_image_file(img_path)
    app.attach(mean_red) # attach the feature to the app when there is some image loaded. it will only happen the first time, and will be ignored after
    target_filename = img_file.replace(".jpg", ".wav")
    app.render_timeline_to_file(my_timeline, target_filename)
    print(f"Saved {target_filename}")
print("Done")

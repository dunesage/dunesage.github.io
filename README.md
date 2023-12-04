
# Alex Akin, AOS C111 Final Project

## Introduction

I obtained a dataset with the title [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) from the data science platform Kaggle. The original source of the images in the dataset is the Earth observation company [Planet Labs](https://www.planet.com/).

## Data

A description of the data is given by its curator, [rhammell](https://www.kaggle.com/rhammell): 

> The dataset consists of images extracted from Planet satellite imagery collected over the San Francisco Bay and San Pedro Bay areas of California. It includes 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification. Images were derived from PlanetScope full-frame visual scene products, which are orthorectified to a 3-meter pixel size.

> The "ship" class includes 1000 images. Images in this class are centered on the body of a single ship. Ships of different sizes, orientations, and atmospheric collection conditions are included. 

> The "no-ship" class includes 3000 images. A third of these are a random sampling of different land cover features - water, vegetation, bare earth, buildings, etc. - that do not include any portion of a ship. The next third are "partial ships" that contain only a portion of a ship, but not enough to meet the full definition of the "ship" class. The last third are images that have previously been mislabeled by machine learning models, typically caused by bright pixels or strong linear features.


I began by loading in the image data and labels:

```python 
def load_images_and_labels(folder_path):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            img = io.imread(image_path)

            # Label the image based on the filename
            label = 'ship' if filename.startswith('1') else 'noship'

            # Append the image and label to the lists
            images.append(img)
            labels.append(label)

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load images and labels
images, labels = load_images_and_labels(folder_path)
```

## Modeling

## Results

## Discussion

## Conclusion

## References

<img align="left" width="800" height="800" src="/Images/visualization_1.png">



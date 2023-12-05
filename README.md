
# Alex Akin, AOS C111 Final Project

## Introduction

Remote sensing from space has grown rapidly in the past decade. The field began in earnest with the launch of Landsat 1, the first Earth observation satellite, by NASA in 1972. Now, publicly-traded companies like [Planet Labs](https://www.planet.com/) are able to image the entire land surface of the Earth daily at unprecedented spatial scales, leading to a massive increase in the amount of data avaliable. Planet maintains an image archive of 50 petabytes that have been collected by its over 200 satellites in orbit, [some having up to 50cm resolution](https://www.planet.com/products/hi-res-monitoring/).

## Data

I obtained a dataset with the title [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) from the data science platform Kaggle. The curator of the data is the user [rhammell](https://www.kaggle.com/rhammell), who extracted "images...from Planet satellite imagery collected over the San Francisco Bay and San Pedro Bay areas of California" 

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

[Tatem, A. J., Goetz, S. J., & Hay, S. I. (2008). Fifty Years of Earth Observation Satellites: Views from above have lead to countless advances on the ground in both scientific knowledge and daily life. American Scientist, 96(5), 390–398. https://doi.org/10.1511/2008.74.390](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2690060/)

<img align="left" width="800" height="800" src="/Images/visualization_1.png">



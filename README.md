
# Alex Akin, AOS C111 Final Project

## Introduction

Remote sensing from space began in earnest with the launch of Landsat 1, the first Earth observation satellite, by NASA in 1972. The field has grown rapidly in recent years. Now, publicly-traded companies like [Planet Labs](https://www.planet.com/) are able to image the entire land surface of the Earth daily at unprecedented spatial scales, leading to a massive increase in the amount of data avaliable. Planet maintains an image archive of 50 petabytes and operates 200 satellites, [some having up to 50cm resolution](https://www.planet.com/products/hi-res-monitoring/). 

The amount of satellite imagery collected by Earth observation satellites has become too large for careful human review. The application of machine learning models can be a helpful tool to apply in the effort to classify images into useful categories.

## Data

I obtained a dataset with the title [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) from the data science platform Kaggle. The curator of the data is the user [rhammell](https://www.kaggle.com/rhammell), who shared 4000 80x80 RGB images "extracted from Planet satellite imagery over the San Francisco Bay and San Pedro Bay areas of California." Of these 4000 images, 1000 were labeled as 'ship' and 3000 as 'no-ship'. The particulars of each class are described below by [rhammell](https://www.kaggle.com/rhammell):

> **'ship':** Images in this class are centered on the body of a single ship. Ships of different sizes, orientations, and atmospheric collection conditions are included. 

> **'no-ship':** A third of these are a random sampling of different land cover features - water, vegetation, bare earth, buildings, etc. - that do not include any portion of a ship. The next third are "partial ships" that contain only a portion of a ship, but not enough to meet the full definition of the "ship" class. The last third are images that have previously been mislabeled by machine learning models, typically caused by bright pixels or strong linear features.

![RGB](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/rgb_ships.png)

## Modeling

I applied a random forest model to the dataset. First, I found the best hyperparameters using `RandomizedSearchCV`, which turned out to be 

`n_estimators=142, max_depth=16, min_samples_leaf=8` with `class_weight='balanced'`

Then, I ran the model, achieving a test accuracy of 96.25% and a training accuracy of 98.97%.

Plotted below is the confusion matrix when the model is run on the test data.

![Confusion Matrix](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/confusion_matrix.png)



## Results

## Discussion

## Conclusion

## References

[Tatem, A. J., Goetz, S. J., & Hay, S. I. (2008). Fifty Years of Earth Observation Satellites: Views from above have lead to countless advances on the ground in both scientific knowledge and daily life. American Scientist, 96(5), 390–398. https://doi.org/10.1511/2008.74.390](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2690060/)


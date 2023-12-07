# Random Forest Ship Detection
**Final Project**

Atmospheric and Oceanic Sciences C111: Introduction to Machine Learning for the Physical Sciences

By Alex Akin

## Introduction

Remote sensing from space began in earnest with the launch of Landsat 1, the first Earth observation satellite, by NASA in 1972 (See [^1] for further reading on the history of remote sensing). The field has grown rapidly in recent years. Now, publicly-traded companies like [Planet Labs](https://www.planet.com/) are able to image the entire land surface of the Earth daily at unprecedented spatial scales, leading to a massive increase in the amount of data avaliable. Planet maintains an image archive of 50 petabytes and operates 200 satellites, [some having up to 50cm resolution](https://www.planet.com/products/hi-res-monitoring/). 

The amount of satellite imagery collected by Earth observation satellites has become too large for careful human review. The application of machine learning models can be a helpful tool to apply in the effort to classify images into useful categories. I use a ***random forest of decision trees*** to classify Planet imagery in order to investigate the opportunities and challenges of this tool from a scientific perspective.

In relation to the wider world of machine learning models, a random forest is an ensemble method that improves upon the performance of an individual decision tree. As my data set was labeled, this is considered supervised learning. Specifically, I selected a binary classification tree architecture, meaning that my decision trees returns either a 0 ('noship') or 1 ('ship'). I built it using scikit-learn's `RandomForestClassifier` implementation. 
![scikitlearn](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/ml_model.png)

## Data

I obtained a dataset with the title [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) from the data science platform Kaggle. The data was posted by the user [rhammell](https://www.kaggle.com/rhammell), who shared 4000 80x80 RGB images "extracted from Planet satellite imagery over the San Francisco Bay and San Pedro Bay areas of California." Of these 4000 images, 1000 were labeled as 'ship' and 3000 as 'no-ship'. The particulars of each class are described below by the curator of the data set:

> **'ship':** Images in this class are centered on the body of a single ship. Ships of different sizes, orientations, and atmospheric collection conditions are included. 

> **'no-ship':** A third of these are a random sampling of different land cover features - water, vegetation, bare earth, buildings, etc. - that do not include any portion of a ship. The next third are "partial ships" that contain only a portion of a ship, but not enough to meet the full definition of the "ship" class. The last third are images that have previously been mislabeled by machine learning models, typically caused by bright pixels or strong linear features.

To illustrate the differences between these two classes of image, I plotted the average red, green, and blue values for the two classes of image (recall that [RGB values](https://en.wikipedia.org/wiki/RGB_color_model) range from 0-255) over an 80x80 grid:

![RGB](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/rgb.png)
#### Figure 1: Average RGB values

We can clearly see the distinction between the 'ship and 'no-ship' classes

## Modeling

I applied a random forest model to the dataset. First, I experimented with hyperparameters using `RandomizedSearchCV`, and landed on these:

`n_estimators=300, max_depth=25, min_samples_leaf=5` with `class_weight='balanced'`

Then, I ran the model, achieving a test accuracy of 96.50% and a training accuracy of 99.47%.


#### Figure 2:

Plotted below is the confusion matrix for the model and the ROC and Precision-Recall curves:

![CM/FI](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/cm_fi.png)
#### Figure 3: Confusion Matrix and Feature Importance

![ROC/REC](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/roc_rec.png)
#### Figure 4: ROC and Precision-Recall Curves

## Results

Finally, I used my model to classify ships with seven of the provided scenes. This was included as a way to visualize the performance of the model as it is applied across a satellite image of a larger area. 

![Results](https://raw.githubusercontent.com/dunesage/dunesage.github.io/main/Images/ship_detection.png)
#### Figure 6:

#### Figure 7:

## Discussion

## Conclusion

## References

[^1]: [Tatem, A. J., Goetz, S. J., & Hay, S. I. (2008). Fifty Years of Earth Observation Satellites: Views from above have lead to countless advances on the ground in both scientific knowledge and daily life. American Scientist, 96(5), 390–398. https://doi.org/10.1511/2008.74.390](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2690060/)
[^2]: 


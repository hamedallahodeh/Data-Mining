# Mushroom Classification Using Data Mining

**Student:** Hamedallah Anwer Hamedallah Issa

**Student ID:** 320220603007

**Supervisor:** Dr. Muhammad Al-Musaideen

---

## 1. What is the project about?

In this project we try to predict if a mushroom is **edible** or **poisonous** using data mining. We took a dataset that has the mushroom's physical features (like the cap, the gills and the stem), then we cleaned it, picked the important features, and trained a decision tree to do the classification.

I think this is a good problem for data mining because a wrong prediction is dangerous, so we really care about the accuracy and the recall.

## 2. The dataset

- File name: `cleaned_secondary_data Final.csv`
- It has **61,070 rows** and **21 columns** at the start.
- The target column is `class`: `p` means poisonous and `e` means edible.
- The other columns are things like `cap-diameter`, `cap-shape`, `cap-surface`, `cap-color`, `gill-color`, `stem-height`, `stem-width`, `stem-color`, `ring-type`, `habitat` and `season`.

## 3. Preprocessing (what we did to the data)

### 3.1 Missing values

First I checked how many missing values each column has. Some columns had too many:

| Column | Missing values |
|---|---|
| gill-spacing | 25,064 |
| cap-surface | 14,120 |
| gill-attachment | 9,884 |

So we did two things:

1. **Dropped** the columns that are mostly empty: `gill-spacing`, `stem-root`, `stem-surface`, `veil-type`, `veil-color` and `spore-print-color`. Filling them would mean making up too much data.
2. For the columns with fewer missing values (`cap-surface`, `gill-attachment`, `ring-type`) we filled the empty cells with the **mode** (the most common value). After that I checked again and there were 0 missing values.

### 3.2 Encoding

The decision tree needs numbers, not letters, so:

- `class` was changed to binary: **1 = poisonous, 0 = edible**.
- All the other text columns (`cap-shape`, `cap-surface`, `cap-color`, `gill-color`, `stem-color`, `ring-type`, `habitat`, `season`, etc.) were converted using `LabelEncoder`.

### 3.3 Scaling

- `stem-height` was normalized using **Z-score**.
- `cap-diameter` and `stem-width` were multiplied by 100 and turned into integers.

## 4. Feature selection

After the cleaning, I trained a decision tree on the training data (80% train / 20% test, `random_state=42`) and used `SelectFromModel` to see which features are important.

It selected these 6 features:

- `cap-surface`
- `gill-attachment`
- `gill-color`
- `stem-height`
- `stem-width`
- `stem-color`

We then dropped the features that were not selected: `cap-diameter`, `cap-shape`, `cap-color`, `does-bruise-or-bleed`, `has-ring`, `habitat` and `season`. We kept `ring-type` in the data too, so the final dataset has the class + 7 features. The cleaned data was saved as `MushroomDataCleand.csv`.

## 5. Information Gain

To double check the features, I calculated the information gain for each one:

| Feature | Information Gain |
|---|---|
| stem-width | 0.1005 |
| stem-height | 0.0598 |
| stem-color | 0.0420 |
| ring-type | 0.0273 |
| cap-surface | 0.0260 |
| gill-attachment | 0.0254 |
| gill-color | 0.0187 |

The most important feature is **stem-width**, then **stem-height**. The gill features had the lowest gain.

## 6. Model and results

We used a **Decision Tree Classifier** from sklearn. The first model, trained on all the features, gave these results on the test set:

| Metric | Result |
|---|---|
| Accuracy | 0.9929 (about 99.3%) |
| Precision | 0.9937 |
| Recall | 0.9936 |

These are very high. The recall is important here because it tells us how many of the poisonous mushrooms the model actually caught.

After the feature selection we also trained a smaller tree with `max_depth=3` on the cleaned data and plotted it, so it is easy to read and explain how the tree takes its decisions. (In the notebook I only plotted it, I didn't print its metrics, so I'm not writing numbers for it here.)

## 7. Conclusion

- Cleaning the data (dropping the empty columns and using the mode for the rest) made the dataset ready for the model.
- Only a few features (mostly the stem ones) carry the most information about whether a mushroom is poisonous.
- A simple decision tree reached around 99% accuracy, so it works very well for this dataset.
- One thing to be careful about: the accuracy is very high, so it could be partly because the dataset has many similar rows. It would be better to test it on new data too.

## 8. Tools I used

Python, pandas, numpy, scikit-learn, scipy, seaborn and matplotlib (in a Jupyter notebook).

# Review Ratings Prediction

## Task
In this project we will observe the power of Transformer-based models. 
The task is to create an NLP model that would be able to predict rating
score (`stars`) users assign to restaurants based on their comments.

## Data
We'll use YELP reviews for this task. Original Yelp Open Dataset is available 
[here](https://www.yelp.com/dataset).

In order to reduce the training time and avoid using 
expensive and often unavailable hardware, we'll extract the first 100,000 records and
split them into training, validation and test sets for development and final model 
evaluation.  

## Repository Structure
* `data` - contains the data we're using for dev and testing purposes.
* `models` - models definitions.
* `notebooks` - auxiliary notebooks, such as data exploration and example of 
the model usage.
* `utils` - helpers and supplementary methods, such as first N records extraction from 
the original dataset.
* `main.py` - default root of the project, not used at the moment.
* `README.md` - the doc you're reading :)
* `requirements.txt` - project dependencies.

## Model



## ToDo
* Add other models for comparison, e.g. Bag-of-Words, RNNs.

import h2o
from h2o.automl import H2OAutoML
# initialize an H2O cluster
# import your data into an H2OFrame
h2o.init()
data = h2o.import_file('./2011_rankings_v2.csv')

y = 'rank'

# initialize an H2O cluster


# import your data into an H2OFrame
#data = h2o.import_file('path/to/your/data.csv')

# split the data into training and validation sets
train, valid = data.split_frame(ratios=[0.8])

# define the response variable and predictor variables
#y = 'target_column'
x = [col for col in data.columns if col != y]

# create an instance of H2OAutoML and start the training process
aml = H2OAutoML(max_models=5, seed=1, include_algos=["XGBoost"])
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

# view the leaderboard to see the performance of the trained models
leaderboard = aml.leaderboard
print(leaderboard)
sample_data = h2o.H2OFrame({
"rank_order": 14,
"scores_overall": 84.9,
"scores_overall_rank": 15,
"scores_teaching": 81.2,
"scores_teaching_rank": 14,
"scores_international_outlook": 66.4,
"scores_international_outlook_rank": 75,
"scores_industry_income": 34.6,
"scores_industry_income_rank": 104,
"scores_research": 88.8,
"scores_research_rank": 15,
"scores_citations": 88.1,
"scores_citations_rank":45 
})
#1.30755
predictions = aml.leader.predict(sample_data)
print("predictions############33")
print(predictions)

# export the best model offline
model_path = h2o.save_model(aml.leader, path='/home/metis/Downloads/')
print(f'Saved model to {model_path}')

# shut down the H2O cluster
h2o.cluster().shutdown()
# use the best model to make predictions on new data
# best_model = aml.leader
# predictions = best_model.predict(data)

# # shut down the H2O cluster
# h2o.cluster().shutdown()

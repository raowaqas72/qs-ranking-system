import h2o

# initialize an H2O cluster
h2o.init()

# load the saved model from the file path
model_path = './XGBoost_3_AutoML_1_20230410_114821'
model = h2o.load_model(model_path)

# create an H2OFrame with new data for inference
new_data = h2o.H2OFrame({
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

# use the loaded model to make predictions on the new data
predictions = model.predict(new_data)
print("")
# print the predicted values
print(predictions)

# shut down the H2O cluster
h2o.cluster().shutdown()

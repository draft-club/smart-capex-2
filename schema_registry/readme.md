## Schema.json : (would be copied here from schema_registry)
### dataset_constraint : Constraint for a dataset number of rows 
- min_fraction_threshold : min fraction of the number of actual dataset length compared to previous dataset
- max_fraction_threshold : max fraction of the number of actual dataset length compared to previous dataset
- unicity_features : features used to check for duplicates. could be a list of features or `"*"` to use all the features
- batch_id_column : the column that contain the id of the batch to validate
- joins_constraint : list of dict with 4 keys
  - name : name of the right join table
  - left_on : feature (str) or features ('list') from the left join table
  - right_on : feature (str) or features ('list') from the right join table
  - threshold : the min possible value for the ratio of the size of the join table on the original one

### features_constraint : Constraint related to a given feature
- name : name of the concerned feature
- type : type of the feature : int, float or str
- presence : fraction [0,1] of values that should be present within a column
- distinctness : a tuple of { "condition" : "" , "value" : "" }
    - condition : on of the values "eq" for equal to, "lt" for less than and "gt" for greater than.
    - value : value  [0,1] used with the condition 
- domain : should conform feature type
    - int and float features should have a {"min":"","max":""} domain
    - str features should have a list of possible values as type 
- regex : pattern that the feature should match. in addition to regular regex special characters, we could use # to refer to another feature
          example : "^#year#(0)?#week#$" the #year# would be replaced with the value of the year column before applying the regex 

- drift : a threshold for Tchebychev distance between the current batch and the previous one 
- outliers : a float value k that define the number of stds around the means that define the non outlier domain 
              non outlier domain : [mean - k*std , mean + k*std]
              
- highest_frequency_threshold : the max frequency that a value could have within column. it could be an integer (max allowed number of repetitions) or a float (the ratio within non-missing values) 
- mapped_to_only_one : the feature or the features that should always take the same value for a given value of the concerned feature 


### slices constraints : to define constraints for a slice of the data
in this entry we should define : 
- slicing column : column on which we are going to slice 
- values to take : array. keep the rows where slicing column is in values to take
- values to drop : array. drop the rows where slicing column is in values to drop
- schema : an embedded schema with all the constraints to validate for the resulting slice of data.

### custom_constraint : More complicated constraint that are very specific to the SmartCapex context 
- max_new_cells : maximum number of allowed new cells compared to previous dataset
- max_disappeared cells : maximum number of allowed disappeared cells compared to previous dataset
- max_cell_per_site : maximum allowed number of cells per site 

     
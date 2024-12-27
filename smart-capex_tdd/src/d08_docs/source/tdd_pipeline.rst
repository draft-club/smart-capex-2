TDD Pipeline
============

.. automodule:: src.tdd_pipeline
   :members:
   :undoc-members:
   :show-inheritance:


Fonctions utilisées dans TDD Pipeline
=====================================

**Fonctions utilisées dans preprocessing_pipeline()**

.. autofunction:: src.d02_preprocessing.OMA.read_process_oss_counter.preprocess_oss_weekly_from_capacity

.. autofunction:: src.d02_preprocessing.OMA.preprocessing_hourly_oss.preprocessing_file_all

---------------------------------------------------------

**Fonctions utilisées dans conversion_rate_pipeline()**

.. autofunction:: src.d02_preprocessing.conversion_rate.compute_conversion_rate
.. autofunction:: src.d02_preprocessing.conversion_rate.change_lte_forecasted
.. autofunction:: src.d02_preprocessing.conversion_rate.post_process_lte

---------------------------------------------------------

**Fonctions utilisées dans train_densification_model_pipeline()**

.. autofunction:: src.d03_processing.OMA.technical_modules.train_densification_impact_model.compute_date_deployment
.. autofunction:: src.d03_processing.OMA.technical_modules.train_densification_impact_model.pipeline_train_model_newsite_deployment
.. autofunction:: src.d03_processing.OMA.technical_modules.traffic_by_region.train_regional_model
.. autofunction:: src.d03_processing.OMA.new_site_modules.train_new_site.train_new_site_data
.. autofunction:: src.d03_processing.OMA.new_site_modules.train_new_site.train_new_site_voice

---------------------------------------------------------

**Fonctions utilisées dans get_randim_densification_result()**

.. autoclass:: src.d04_randim.call_randim.ApiRandim

---------------------------------------------------------


**Fonctions utilisées dans apply_densification_model_pipeline()**

.. autofunction:: src.d03_processing.OMA.technical_modules.apply_traffic_gain_densification.pipeline_apply_traffic_gain_densification
.. autofunction:: src.d03_processing.OMA.technical_modules.traffic_by_region.predict_improvement_traffic_trend_kpis
.. autofunction:: src.d03_processing.OMA.technical_modules.traffic_by_region.split_data_traffic

---------------------------------------------------------

**Fonctions utilisées dans densification_to_economical_pipeline()**

.. autofunction:: src.d03_processing.OMA.technical_modules.arpu_quantification.compute_revenues_per_site
.. autofunction:: src.d03_processing.OMA.technical_modules.arpu_quantification.compute_increase_of_arpu_by_the_upgrade

---------------------------------------------------------

**Fonctions utilisées dans density_economical_pipeline()**

.. autofunction:: src.d03_processing.OMA.economical_modules.gross_margin_quantification.compute_site_margin
.. autofunction:: src.d03_processing.OMA.economical_modules.gross_margin_quantification.compute_increase_of_yearly_site_margin
.. autofunction:: src.d03_processing.OMA.economical_modules.npv_computation.compute_npv

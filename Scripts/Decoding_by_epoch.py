#!/usr/bin/env python
# coding: utf-8

# ## Clusterless Decoding Analysis W-Track | Compute decoded position and distance metrics based on CA1 Marks and associated body position | Inputs: Marks, Posdlc, Task, Tetinfo | https://github.com/edeno/pose_analysis

# ### Import pre-existing libraries 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import seaborn as sns
import logging

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
sns.set_context("talk")


# ## Load Data 
# ### Specifiy animal, day, epoch and body position estimate to be used to encode pos-mark relationship. 

# In[ ]:



from src.load_data import load_data

epoch_key = ("Jaq", 3,16) # animal, day, epoch

data=load_data(epoch_key,
              position_to_linearize=['nose_x', 'nose_y'],
              max_distance_from_well=30,
              min_distance_traveled=50,
              )


# In[ ]:


# plots the linearised position coloured by segments on the w-track 

fig, ax = plt.subplots(figsize=(30, 10))

for edge_label, df in data['position_info'].groupby('track_segment_id'):
    ax.scatter(df.index / np.timedelta64(1, 's'), df.linear_position, s=1)
    
ax.set_ylabel('Position [cm]')
ax.set_xlabel('Time [s]');


# In[ ]:


# selects periods to be included for analysis, such as velocity>4cm/sec or inbound and outbound trials 

from src.parameters import EDGE_ORDER, EDGE_SPACING, ANIMALS
from src.load_data import make_track_graph

track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)
is_running = np.abs(data["position_info"].nose_vel) > 0
#is_running = np.abs(data["position_info"].forepawR_vel) > 4
#is_outbound = data["position_info"].task == "Outbound"


# In[ ]:


# plots trajectories classified as inbound or outbound. Good sanity checks for if the times are correctly defined. Atm does NOT use DIOs, can be implemented in the future

fig, ax = plt.subplots(1, 1, figsize=(30, 10))
ax.scatter(
        data["position_info"].index / np.timedelta64(1, "s"), data["position_info"].linear_position, s=10, color='lightgrey',
    )
for task, df in data["position_info"].groupby("task"):
    ax.scatter(
        df.index / np.timedelta64(1, "s"), df.linear_position, s=10, label=task,
    )

plt.legend()
sns.despine()


# ### Calculate Posterior | Builds the classifier and calculates the posterior estimates for each bin. Default is 5x cross validation. Some concerns if that is appropriate in 15 minute run epochs,but AJ checked that the overal posterior was similar in 2x,3x, and 5x versions. Maybe stick to 3x for 15 minute data?

# In[ ]:



from replay_trajectory_classification import ClusterlessClassifier
from src.parameters import classifier_parameters, discrete_state_transition

from sklearn.model_selection import KFold
from tqdm.auto import tqdm

cv = KFold()
cv_classifier_clusterless_results = []

for fold_ind, (train, test) in tqdm(enumerate(cv.split(data["position_info"].index))):
    
   # train = train[is_outbound[train].values]
    
    cv_classifier = ClusterlessClassifier(**classifier_parameters)

    cv_classifier.fit(
        position=data["position_info"].iloc[train].linear_position,
        multiunits=data["multiunits"].isel(time=train),
        is_training=is_running.iloc[train],
        track_graph=track_graph,
        center_well_id=center_well_id,
        edge_order=EDGE_ORDER,
        edge_spacing=EDGE_SPACING,
    )
    cv_classifier.discrete_state_transition_ = discrete_state_transition
    logging.info('Predicting posterior...')
    cv_classifier_clusterless_results.append(
        cv_classifier.predict(
            data["multiunits"].isel(time=test),
            time=data["position_info"].iloc[test].index / np.timedelta64(1, "s"),
        )
    )


# In[ ]:


# concatenate cv classifier results 
cv_classifier_clusterless_results = xr.concat(
    cv_classifier_clusterless_results, dim="time"
)
cv_classifier_clusterless_results


# ### Calculate Distance Metrics 

# In[ ]:


# Important calculate distance metrics. Loads Causal Posterior and get_trajectory_data and
# get_distance_metrics to  calculate ahead-behind distance based on body_dir 
## CAUSAL 

from trajectory_analysis_tools import get_distance_metrics, get_trajectory_data

posterior_causal = (cv_classifier_clusterless_results["causal_posterior"]
             .sum("state", skipna=False))

#extracting the peak of the posterior 
trajectory_data = get_trajectory_data(
    posterior=posterior_causal,
    track_graph=track_graph,
    decoder=cv_classifier,
    position_info=data["position_info"],
    direction_variable="body_dir"
)

#
distance_metrics = get_distance_metrics(
    track_graph, *trajectory_data)

ahead_behind_distance_causal = (
    distance_metrics.mental_position_ahead_behind_animal *
    distance_metrics.mental_position_distance_from_animal)


# In[ ]:


# Calculate the corresponding 95% HPD credible interval 
from trajectory_analysis_tools import get_highest_posterior_threshold, get_HPD_spatial_coverage

hpd_threshold_95_causal = get_highest_posterior_threshold(posterior_causal.dropna("position"), coverage=0.95)
spatial_coverage_95_causal = get_HPD_spatial_coverage(posterior_causal, hpd_threshold_95_causal)


# In[ ]:


# Calculate the corresponding 50% HPD credible interval 

from trajectory_analysis_tools import get_highest_posterior_threshold, get_HPD_spatial_coverage

hpd_threshold_50_causal = get_highest_posterior_threshold(posterior_causal.dropna("position"), coverage=0.50)
spatial_coverage_50_causal = get_HPD_spatial_coverage(posterior_causal, hpd_threshold_50_causal)


# In[ ]:


# calculate distance metrics acausal posterior. Loads acausal posterior and 
# distance metrics. ACAUSAL

from trajectory_analysis_tools import get_distance_metrics, get_trajectory_data

posterior_acausal = (cv_classifier_clusterless_results["acausal_posterior"]
             .sum("state", skipna=False))

#extracting the peak of the posterior 
trajectory_data = get_trajectory_data(
    posterior=posterior_acausal,
    track_graph=track_graph,
    decoder=cv_classifier,
    position_info=data["position_info"],
    direction_variable="body_dir"
)

distance_metrics = get_distance_metrics(
    track_graph, *trajectory_data)

ahead_behind_distance_acausal = (
    distance_metrics.mental_position_ahead_behind_animal *
    distance_metrics.mental_position_distance_from_animal)


# In[ ]:


# ACAUSAL 95% CI
from trajectory_analysis_tools import get_highest_posterior_threshold, get_HPD_spatial_coverage

hpd_threshold_95_acausal = get_highest_posterior_threshold(posterior_acausal.dropna("position"), coverage=0.95)
spatial_coverage_95_acausal = get_HPD_spatial_coverage(posterior_acausal, hpd_threshold_95_acausal)


# In[ ]:


# ACAUSAL 50% CI
from trajectory_analysis_tools import get_highest_posterior_threshold, get_HPD_spatial_coverage

hpd_threshold_50_acausal = get_highest_posterior_threshold(posterior_acausal.dropna("position"), coverage=0.50)
spatial_coverage_50_acausal = get_HPD_spatial_coverage(posterior_acausal, hpd_threshold_50_acausal)


# In[ ]:


# WHILE WE ARE AT IT, ALSO A GOOD IDEA TO CALCULATE THE ABSOLUTE DISTANCE. 
# CAUSAL 
from src.analysis import calculate_replay_distance

replay_distance_from_animal_position_causal = calculate_replay_distance(
    posterior=cv_classifier_clusterless_results.causal_posterior.sum('state'),
    track_graph=track_graph,
    decoder=cv_classifier,
    position_2D=data['position_info'].loc[:, ["nose_x", "nose_y"]],
    track_segment_id=data['position_info'].loc[:, ["track_segment_id"]],
)
replay_distance_from_animal_position_causal


# In[ ]:


# WHILE WE ARE AT IT, ALSO A GOOD IDEA TO CALCULATE THE ABSOLUTE DISTANCE. 
# ACAUSAL 
from src.analysis import calculate_replay_distance

replay_distance_from_animal_position_acausal = calculate_replay_distance(
    posterior=cv_classifier_clusterless_results.acausal_posterior.sum('state'),
    track_graph=track_graph,
    decoder=cv_classifier,
    position_2D=data['position_info'].loc[:, ["nose_x", "nose_y"]],
    track_segment_id=data['position_info'].loc[:, ["track_segment_id"]],
)
replay_distance_from_animal_position_acausal


# ### Save the distance and CI values with the classifier results 

# In[ ]:



cv_classifier_clusterless_results['abs_distance_from_animal_position_causal'] = (('time'), replay_distance_from_animal_position_causal)
cv_classifier_clusterless_results['abs_distance_from_animal_position_acausal'] = (('time'), replay_distance_from_animal_position_acausal)

# maybe this will works and we can save both distances 
cv_classifier_clusterless_results['rel_distance_from_animal_position_causal'] = (('time'), ahead_behind_distance_causal)
cv_classifier_clusterless_results['rel_distance_from_animal_position_acausal'] = (('time'), ahead_behind_distance_acausal)

# get HPD estimate of the distance associated
cv_classifier_clusterless_results['hpd_threshold_95_causal'] = (('time'), hpd_threshold_95_causal)
cv_classifier_clusterless_results['hpd_threshold_50_causal'] = (('time'), hpd_threshold_50_causal)
cv_classifier_clusterless_results['hpd_threshold_95_acausal'] = (('time'), hpd_threshold_95_acausal)
cv_classifier_clusterless_results['hpd_threshold_50_acausal'] = (('time'), hpd_threshold_50_acausal)

# get CI of the distance associated
cv_classifier_clusterless_results['credible_interval_95_causal'] = (('time'), spatial_coverage_95_causal)
cv_classifier_clusterless_results['credible_interval_50_causal'] = (('time'), spatial_coverage_50_causal)
cv_classifier_clusterless_results['credible_interval_95_acausal'] = (('time'), spatial_coverage_95_acausal)
cv_classifier_clusterless_results['credible_interval_50_acausal'] = (('time'), spatial_coverage_50_acausal)


# In[ ]:


cv_classifier_clusterless_results


# ### Save the results and model

# In[ ]:


# save the results as .nc format. ncread matlab can read these
cv_classifier_clusterless_results.to_netcdf(
   f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}_cv_classifier_clusterless_vel_0_nose_alltime_results.nc"
)


# In[ ]:


# good idea to save the model so that we dont have to run the entire model again to calculate disatnces 
cv_classifier.save_model(f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}_cv_classifier_clusterless_nose.pkl")


# In[8]:


data['position_info'].to_csv(f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}_position_info_nose.csv")


# In[ ]:




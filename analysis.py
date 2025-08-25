# -*- coding: utf-8 -*-
"""
@author: Laura
"""

#%% import of important packages

import pandas as pd
import os  # to import data
import numpy as np
import ast # to convert list representation to list
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from geoparser import Geoparser
import geopandas
from sklearn.neighbors import KernelDensity
import cartopy
import cartopy.crs as ccrs
import seaborn as sns

os.chdir(r"C:\Users\Startklar\OneDrive\Dokumente\Master\Data")

figure_path = r"C:\Users\Startklar\OneDrive\Dokumente\Master\Output"

#%% import of data

# 90% corpus
corpus_uncoded = pd.read_excel('90p_uncoded_data.xlsx')
corpus_uncoded_pred = pd.read_excel('corpus_uncoded_predictions.xlsx')
corpus_90p = pd.merge(corpus_uncoded_pred, corpus_uncoded, on='doc_no', how='left')
corpus_90p_subset = corpus_90p[['doc_no', 'pred', 'year', 'disaster', 'Bodytext', 'clause', 'clean_clause', 'date']] 
corpus_90p_subset = corpus_90p_subset.rename(columns={'pred': 'capacity'})

# 10% corpus
corpus_coded = pd.read_excel('10p_coded_data.xlsx')
coded_seg = pd.read_excel('segments_data_cleaned.xlsx')
coded_seg = coded_seg[['doc_no', 'code', 'cleaned_code', 'capacity']]
corpus_10p = pd.merge(coded_seg, corpus_coded, on='doc_no', how='left')
corpus_10p = corpus_10p.rename(columns={'code': 'clause', 'cleaned_code': 'clean_clause'})
corpus_10p_subset = corpus_10p[['doc_no', 'capacity', 'year', 'disaster', 'Bodytext', 'clause', 'clean_clause', 'date']]

corpus_all_cap = pd.concat([corpus_90p_subset, corpus_10p_subset]).reset_index(drop=True)
corpus_all_cap[['disaster']] = corpus_all_cap[['disaster']].map(ast.literal_eval)

corpus_all_doc = corpus_all_cap[['doc_no', 'year', 'disaster', 'Bodytext']].groupby(['doc_no']).agg({'year': 'first','disaster': 'first','Bodytext': 'first'}).reset_index()


#%% functions for temporal analysis

# all charts with Arial
plt.rcParams['font.family'] = 'Arial'

# function to count disaster per year
def plotDisasterOccurrences(df, png_name):
    """
    Splits disaster data by type and year, counts the occurrences, and plots them as a stacked area chart.

    Returns a DataFrame with yearly counts for each disaster type.
    
    """
    df_new = df.copy()
    df_exploded = df_new.explode('disaster')
    df_exploded = df_exploded.reset_index(drop=True)
    df_grouped = df_exploded.groupby(['year', 'disaster']).size().reset_index()
    df_grouped.columns.values[2] = 'no_disaster'
    no_disaster = df_grouped.pivot(index='year', columns='disaster', values='no_disaster').reset_index().fillna(0).astype(int)
    
    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    
    colors = plt.get_cmap('Pastel2').colors

    plt.stackplot(no_disaster['year'], no_disaster['drought'], no_disaster['flood'],
                  no_disaster['heatwave'], no_disaster['hurricane'], no_disaster['storm'],
                  no_disaster['tornado'], no_disaster['wildfire'],
                  labels=['drought', 'flood', 'heatwave', 'hurricane', 'storm', 'tornado', 'wildfire'],
                  colors=colors[:7])
    plt.xlabel('Year')
    plt.ylabel('Number of Disaster Reports')
    plt.legend(loc='upper left')
    plt.xlim(no_disaster['year'].min(), no_disaster['year'].max())
    plt.tight_layout()
    fig.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.show()
    

    return no_disaster

# function to count capacity occurrences per year
def plotCapacityOccurrences(df, png_name):
    """
    Groups disaster data by capacity and year, counts the occurrences, and plots them as a line chart.

    Returns a DataFrame with yearly counts for each capacity type.
    
    """
    df_grouped = df.groupby(['year', 'capacity']).size().reset_index()
    df_grouped.columns.values[2] = 'no_disaster'
    no_capacity = df_grouped.pivot(index='year', columns='capacity', values='no_disaster').reset_index().fillna(0).astype(int).set_index('year')
    no_capacity = no_capacity[['preventive', 'anticipative', 'absorptive', 'adaptive', 'transformative']]
    
    fig, ax = plt.subplots(figsize=(10,6))
    no_capacity.plot(cmap='Pastel2', linewidth=2, ax=ax) #'viridis_r'
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Resilience Capacity Mentions')
    ax.legend(loc='upper left')
    plt.xlim(no_capacity.index.min(), no_capacity.index.max())
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.show()
    
    return no_capacity


#%% functions for location extraction

# extract locations for each unique document number
def extractLocations(df, geoparser, threshold=0.7):
    """
    Extracts geographic locations from text using a geoparser and filters them by confidence score.
    Groups documents by ID, parses text to detect toponyms, and collects the location name and its corresponding country and coordinates if the 
    confidence threshold is reached. Creates a DataFrame with all toponyms and only unique toponyms per document.

    Returns the geoparser output, a DataFrame with all extracted locations, and a DataFrame with unique locations per document.
    
    """
    unique_doc = df.groupby('doc_no')['Bodytext'].first().reset_index()
    article_list = unique_doc['Bodytext'].tolist()
    locations_gp = geoparser.parse(article_list)
    data_all = []
    data_unique = []
    for doc_id, doc in zip(unique_doc['doc_no'], locations_gp):
        filtered_toponyms = list()
        filtered_countries = list()
        filtered_coordinates = list()
        for toponym, location in zip(doc.toponyms, doc.locations):
            if toponym.score and toponym.score > threshold:
                if location:
                    if location.get('country_name') and location.get('latitude') is not None and location.get('longitude') is not None:
                        filtered_toponyms.append(str(toponym))
                        filtered_countries.append(location['country_name'])
                        filtered_coordinates.append((location['latitude'], location['longitude']))
        
        if filtered_toponyms: 
            data_all.append({
                'doc_no': doc_id,
                'location': filtered_toponyms,
                'country': filtered_countries,
                'coordinates': filtered_coordinates
                })
            
            data_unique.append({
                'doc_no': doc_id,
                'location': set(filtered_toponyms),
                'country': set(filtered_countries),
                'coordinates': set(filtered_coordinates)
                })
    
    df_locations_all = pd.DataFrame(data_all)
    df_locations_unique = pd.DataFrame(data_unique)
    return locations_gp, df_locations_all, df_locations_unique

# create instance of Geoparser class
geoparser = Geoparser(spacy_model='en_core_web_trf', transformer_model='dguzh/geo-all-distilroberta-v1')

locations, df_locations_all, df_locations_unique = extractLocations(corpus_all_doc, geoparser)

df_locations_all.to_excel('corpus_locations_all.xlsx', index=False)
df_locations_unique.to_excel('corpus_locations_unique.xlsx', index=False)


#%% functions for spatial analysis

# function to create gdf
def createGeodataframe(df):
    """
    Converts a DataFrame with coordinates into a GeoDataFrame in Mollweide projection.
    Handles cases where coordinates are stored as lists, extracts latitude and longitude, and builds point geometries in WGS84 before reprojecting to Mollweide.

    Returns a GeoDataFrame with document IDs and geometries.
    
    """
    if isinstance(df['coordinates'].iloc[0], list):
        df_new = df.copy()
        df_new = df_new.explode('coordinates')
        df_new = df_new.reset_index(drop=True)
    else:
        df_new = df.copy()
    df_new['Latitude'] = df_new['coordinates'].apply(lambda coord: coord[0] if coord else None)
    df_new['Longitude'] = df_new['coordinates'].apply(lambda coord: coord[1] if coord else None)
    gdf = geopandas.GeoDataFrame(df_new, geometry=geopandas.points_from_xy(df_new.Longitude, df_new.Latitude), crs='EPSG:4326')
    gdf_final = gdf[['doc_no', 'geometry']]
    gdf_mollweide = gdf_final.to_crs('ESRI:54009')
    return gdf_mollweide


# function to create KDE heatmap    
def plotKDE(gdf, png_name):
    """
    Plots a Kernel Density Estimation (KDE) heatmap in a Mollweide projection.
    Takes a GeoDataFrame with point geometries, estimates spatial density using Gaussian KDE, applies a threshold to highlight denser regions, 
    and overlays the results on a world map.

    Displays a heatmap and geographic features.
    
    """
    moll = ccrs.Mollweide()
    
    coordinates = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
    
    # define grid based on dataset extent
    x_min, y_min, x_max, y_max = gdf.total_bounds
    x_grid = np.arange(x_min, x_max, 100000)
    y_grid = np.arange(y_min, y_max, 100000)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    xy_mesh = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
    
    # fit and predict Kernel Density Estimation model
    kde = KernelDensity(bandwidth=200000, kernel='gaussian', metric='euclidean') # bandwidth='silverman',
    kde.fit(coordinates)
    pred = np.exp(kde.score_samples(xy_mesh))
    pred = pred.reshape(x_mesh.shape)
    
    threshold = np.percentile(pred, 70)  
    pred_masked = np.ma.masked_where(pred < threshold, pred)
    
    # normalize values
    vmin = threshold
    vmax = pred_masked.max()
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)
    lognorm = colors.LogNorm(vmin=vmin, vmax=vmax)
       
    # settings for plot
    plt.figure(figsize=(12,6))
    ax = plt.axes(projection=moll)
    ax.set_extent([x_min, x_max, y_min, y_max], crs=moll)
    
    # plot KDE
    ax.contourf(x_mesh, y_mesh, pred_masked, levels=levels, cmap=plt.cm.YlOrRd, norm=lognorm, alpha=0.8, transform=moll)
  
    # add geographic features
    ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='#B1B2B4', linewidth=0.001)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.25)
    
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.show()    
    

# function to determine if article belongs to GN or GS
def categoriseGS(df, list_gs):
    """
    Categorises documents into Global South (GS) or Global North (GN) based on country mentions and a country list.
    Counts number of countries from GS, calculates ratio compared to total countries and assigns an overall region. Creates separate DataFrames for GS and GN records.

    Returns three DataFrames: DataFrame with regions for all documents, DataFrame with GN-specific entries, and DataFrame with GS-specific entries.
    
    """
    df_new = df.copy()
    
    # count GS countries and total countries per document and calculate ratio
    df_new['gs_count'] = df_new['country'].apply(lambda x: sum(country in list_gs for country in x))
    df_new['all_count'] = df_new['country'].apply(lambda x: len(x))
    df_new['gs_ratio'] = df_new['gs_count'] / df_new['all_count']
    df_new['region_overall'] = np.where(df_new['gs_ratio'] >= 0.5, 'GS', 'GN')
    df_explode = df_new.explode(['location', 'country', 'coordinates'])
    df_explode['region'] = np.where(df_explode['country'].isin(list_gs), 'GS', 'GN')
    
    # create subsets
    df_gn = df_explode.loc[(df_explode['region'] == 'GN') & (df_explode['region_overall'] == 'GN')].reset_index(drop=True)
    df_gn_final = df_gn[['doc_no', 'location', 'country', 'coordinates']].drop_duplicates()
    df_gs = df_explode.loc[(df_explode['region'] == 'GS') & (df_explode['region_overall'] == 'GS')].reset_index(drop=True)
    df_gs_final = df_gs[['doc_no', 'location', 'country', 'coordinates']].drop_duplicates()
    df_final = df_explode[['doc_no', 'region_overall']].drop_duplicates()
    
    return df_final, df_gn_final, df_gs_final
    

# list of Global South countries included in thesis    
global_south_countries = ["Afghanistan", "Algeria", "Angola", "Antigua and Barbuda", "Argentina", "Azerbaijan", "Bahamas", "Bahrain",
                          "Bangladesh", "Barbados", "Belize", "Benin", "Bhutan", "Bolivia", "Botswana", "Brazil", "Brunei Darussalam", 
                          "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Chile",
                          "China", "Colombia", "Comoros", "Democratic Republic of the Congo", "Costa Rica", "Ivory Coast", "Cuba", 
                          "North Korea", "Republic of the Congo", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", 
                          "El Salvador", "Equatorial Guinea", "Eritrea", "Ethiopia", "Fiji", "Gabon", "Gambia", "Ghana", 
                          "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "India", "Indonesia", 
                          "Iran", "Iraq", "Jamaica", "Jordan", "Kenya", "Kiribati", "Kuwait", "Laos", "Lebanon", "Lesotho", "Liberia", 
                          "Libya", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Marshall Islands", "Mauritania", "Mauritius", 
                          "Micronesia", "Mongolia", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Nicaragua", 
                          "Niger", "Nigeria", "Oman", "Pakistan", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", 
                          "Qatar", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", 
                          "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Seychelles", "Sierra Leone", "Singapore", "Solomon Islands", 
                          "Somalia", "South Africa", "South Sudan", "Sri Lanka", "Palestinian Territory", "Sudan", "Suriname", 
                          "Swaziland", "Syria", "Tajikistan", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", 
                          "Tunisia", "Turkmenistan", "Uganda", "United Arab Emirates", "Tanzania", "Uruguay", "Vanuatu", "Venezuela", 
                          "Vietnam", "Yemen", "Zambia", "Zimbabwe"]


# function to determine disaster typology for each article
def categoriseDisasterType(df, list_slow_onset):
    """
    Categorises disasters into slow-onset or sudden-onset based on a keyword list.
    Checks if article talks about a drought, and if so, categorises it as slow-onset.

    Returns three DataFrames: Dataframe with disaster type for all documents, Dataframe with slow-onset-specific entries and DataFrame with sudden-onset-specific entries.
        
    """
    df_new = df.copy()
    
    # assign disaster type based on presence of slow-onset keywords
    df_new['disaster_type'] = np.where(df_new['disaster'].apply(lambda x: any(word in slow_onset for word in x)), 'slow-onset', 'sudden-onset')
    df_explode = df_new[['doc_no', 'location', 'country', 'coordinates', 'disaster_type']].explode(['location', 'country', 'coordinates'])
    
    # create subsets
    df_slow_onset = df_explode.loc[(df_explode['disaster_type'] == 'slow-onset')].reset_index(drop=True).drop_duplicates()
    df_sudden_onset = df_explode.loc[(df_explode['disaster_type'] == 'sudden-onset')].reset_index(drop=True).drop_duplicates()
    df_final = df_explode[['doc_no', 'disaster_type']].drop_duplicates()
    
    return df_final, df_slow_onset, df_sudden_onset


# list of slow-onset disasters included in thesis  
slow_onset = ['drought']


#%% functions for thematic analysis 

# function to plot heat map
def plotHeatMap(df, group_feature, png_name):
    """
    Creates a heatmap showing topic counts by capacity and a chosen grouping feature.
    Groups data by the given feature, capacity, and topic, normalizes counts across grouping feature, and plots heatmap.
    
    """
    # define topic order for consistent sorting
    order_topic = ['DRM', 'Structural Measure', 'Governance & Policy', 'Strategic Planning', 'Risk Awareness', 'Early Warning', 'Scenario Planning', 'Risk Transfer Mechanism', 'Resource Management', 'Operational Adjustment', 'Support', 'Preparation & Response', 'Diversification', 'Learning', 'Incremental Adjustment', 'Institutional Adaptation', 'Livelihood Transformation', 'Technical Innovation', 'Social Transformation', 'Governance Transformation']
    
    # aggregate counts per grouping feature - capacity - topic combination
    df_hmap = df.groupby(by=[group_feature, 'capacity', 'topic']).size().reset_index(name='count')
    df_hmap['topic'] = pd.Categorical(df_hmap['topic'], categories=order_topic, ordered=True)
    df_hmap = df_hmap.sort_values('topic')
    
    pivot_hmap = df_hmap.pivot(index=['topic', 'capacity'], columns=group_feature, values='count').fillna(0)
        
    pivot_hmap_norm = (pivot_hmap - pivot_hmap.min()) / (pivot_hmap.max() - pivot_hmap.min())
    pivot_hmap_norm = pivot_hmap_norm.droplevel('capacity')
    
    # determine where capacity changes for better representation
    capacities = pivot_hmap.index.get_level_values('capacity')
    change_points = np.where(capacities[:-1] != capacities[1:])[0]+1
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(pivot_hmap_norm, cmap='crest', ax=ax)
    ax.hlines(change_points.tolist(), *ax.get_xlim(), linewidth=3.5, color="w")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    # add capacity labels
    change_points_last = np.append(change_points, len(capacities))
    for i in range(len(change_points_last)):
        pos = change_points_last[i]-1
        cap_label = capacities[pos]
        ax.text(-1.5, pos-1, cap_label, va='center', ha='left', fontsize=12, weight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    
    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.show()


#%% prepare datasets for GN & GS and slow- and sudden-onset disasters

# datasets for Global North and Global South
# dataset with all locations
df_locations_all = pd.read_excel('corpus_locations_all.xlsx')
df_locations_all[['location', 'country', 'coordinates']] = df_locations_all[['location', 'country', 'coordinates']].map(ast.literal_eval)

# determine dfs with all locations in Global North and Global South and a df with the dominant region for each doc_no
df_locations_region, df_gn, df_gs = categoriseGS(df_locations_all, global_south_countries)

# dataset with unique locations
df_locations_unique = pd.read_excel('corpus_locations_unique.xlsx')
df_locations_unique[['location', 'country', 'coordinates']] = df_locations_unique[['location', 'country', 'coordinates']].map(ast.literal_eval).map(lambda x: list(x))


# dataset for slow- and sudden-onset disasters
df_locations_disaster = pd.merge(df_locations_all, corpus_all_doc[['doc_no', 'disaster']])

# determine dfs with all locations for slow-onset and sudden-onset disasters and a df with the disaster type for each doc_no
df_locations_disaster_type, df_slow_onset, df_sudden_onset = categoriseDisasterType(df_locations_disaster, slow_onset)

# join information about region and disaster type to capacity df
df_all_information = pd.merge(corpus_all_cap, df_locations_region, how='left', on='doc_no')
df_all_information = pd.merge(df_all_information, df_locations_all[['doc_no', 'location', 'country']], how='left', on='doc_no')
df_all_information = pd.merge(df_all_information, df_locations_disaster_type, how='left', on='doc_no')

df_all_information.to_excel('corpus_predictions_additional_info.xlsx', index=False)


#%% general analysis

# import data with manually annotated subcategories
df_subcategories = pd.read_excel('corpus_predictions_additional_info_final.xlsx', sheet_name='final')
df_subcategories_filtered = df_subcategories[df_subcategories['topic']!='-']

# determine how many relevant examples there are in 10p and 90p corpora
corpus_10p_docno = corpus_10p['doc_no'].unique().tolist()
df_extracted_90p = df_subcategories_filtered[~df_subcategories_filtered['doc_no'].isin(corpus_10p_docno)]['capacity'].value_counts()
df_extracted_10p = df_subcategories_filtered[df_subcategories_filtered['doc_no'].isin(corpus_10p_docno)]['capacity'].value_counts()

# calculate number of segments in relevant articles
doc_counts = corpus_all_cap['doc_no'].value_counts().reset_index()
count_freq = doc_counts['count'].value_counts()
freq_perc = (count_freq / count_freq.sum()) * 100
df_count_freq = pd.DataFrame({'count': count_freq.index, 'num_doc': count_freq.values, 'percentage': freq_perc.values})


#%% temporal analysis

######################################################
#                   global analysis
######################################################

# plot disasters per year
no_disaster = plotDisasterOccurrences(corpus_all_doc, os.path.join(figure_path, "no_disaster_general.png"))


# plot capacity occurences per year
no_capacity = plotCapacityOccurrences(corpus_all_cap, os.path.join(figure_path, "no_capacity_general_wo_disaster_no.png"))


# special plot including general number of disaster articles
fig, ax = plt.subplots(figsize=(10,6))
no_capacity.plot(cmap='Pastel2', linewidth=2, ax=ax)

no_article = corpus_all_doc['year'].value_counts().reset_index().sort_values(by=['year'])
no_article_plot = no_article.set_index('year')
no_article_plot['count'].plot(ax=ax, color='black', linestyle='--', linewidth=2, label='total articles')

plt.xlim(no_capacity.index.min(), no_capacity.index.max())
ax.set_ylim(bottom=0)
plt.xlabel('Year')
plt.ylabel('Number of Resilience Capacity Mentions')
plt.legend(loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(figure_path, "no_capacity_general.png"), dpi=300, bbox_inches="tight")
plt.show()


###################################################### 
#     analysis of Global North and Global South
######################################################

doc_gn = df_gn['doc_no'].unique()

# create plots with number of capacities per year for regions
df_gn_capacity = corpus_all_cap.loc[corpus_all_cap['doc_no'].isin(doc_gn)]
plotCapacityOccurrences(df_gn_capacity, os.path.join(figure_path, "no_capacity_gn.png"))

df_gs_capacity = corpus_all_cap.loc[~corpus_all_cap['doc_no'].isin(doc_gn)]
plotCapacityOccurrences(df_gs_capacity, os.path.join(figure_path, "no_capacity_gs.png"))


######################################################
#   analysis of slow-onset and sudden-onset disaster  
######################################################

# create plots with number of capacities per year for disaster typologies
df_slow_onset_capacity = corpus_all_cap[corpus_all_cap['disaster'].apply(lambda x: any(word in slow_onset for word in x))]
plotCapacityOccurrences(df_slow_onset_capacity, os.path.join(figure_path, "no_capacity_slow_onset.png"))

df_sudden_onset_capacity = corpus_all_cap[~corpus_all_cap['disaster'].apply(lambda x: any(word in slow_onset for word in x))]
plotCapacityOccurrences(df_sudden_onset_capacity, os.path.join(figure_path, "no_capacity_sudden_onset.png"))


#%% spatial analysis

######################################################
#                   global analysis
######################################################

# global analysis
gdf_mollweide_all = createGeodataframe(df_locations_all)
gdf_mollweide_unique = createGeodataframe(df_locations_unique)

plotKDE(gdf_mollweide_unique, os.path.join(figure_path, "KDE_global.png"))

# determine number of unique toponyms and countries
print("Number of unique place names: {}".format(df_locations_unique['location'].explode().nunique()))
print("Number of unique country names: {}".format(df_locations_unique['country'].explode().nunique()))
print(df_locations_unique['country'].explode().value_counts().head(50))
print(df_locations_unique['location'].explode().value_counts().head(50))

# determine which locations are dominant
df_locations_wo_country = df_locations_all[['doc_no', 'location', 'country']].explode(['location', 'country']).drop_duplicates()
df_locations_wo_country = df_locations_wo_country.loc[df_locations_wo_country['country']!=df_locations_wo_country['location']]
print(df_locations_wo_country[['location', 'country']].value_counts().head(50))


###################################################### 
#     analysis of Global North and Global South
######################################################

# convert df to gdf and plot KDE map
gdf_mollweide_gn = createGeodataframe(df_gn)
gdf_mollweide_gs = createGeodataframe(df_gs)

plotKDE(gdf_mollweide_gn, os.path.join(figure_path, "KDE_GN.png"))
plotKDE(gdf_mollweide_gs, os.path.join(figure_path, "KDE_GS.png"))

# determine which locations are dominant in the Global North
df_gn_wo_country = df_gn.loc[df_gn['country']!=df_gn['location']]
print(df_gn_wo_country[['location', 'country']].value_counts().head(60))
print(df_gn[['doc_no', 'country']].drop_duplicates()['country'].value_counts().head(50))
print(df_gn['location'].value_counts().head(50))

# determine which locations are dominant in the Global South
df_gs_wo_country = df_gs.loc[df_gs['country']!=df_gs['location']]
print(df_gs_wo_country[['location', 'country']].value_counts().head(60))
print(df_gs[['doc_no', 'country']].drop_duplicates()['country'].value_counts().head(50))
print(df_gs['location'].value_counts().head(50))


######################################################
#   analysis of slow-onset and sudden-onset disaster  
######################################################

# convert df to gdf and plot KDE map
gdf_slow_onset = createGeodataframe(df_slow_onset)
gdf_sudden_onset = createGeodataframe(df_sudden_onset)

plotKDE(gdf_slow_onset, os.path.join(figure_path, "KDE_slow_onset.png"))
plotKDE(gdf_sudden_onset, os.path.join(figure_path, "KDE_sudden_onset.png"))

# determine which locations are dominant for slow-onset 
df_slow_wo_country = df_slow_onset[['location', 'country']].explode(['location', 'country'])
df_slow_wo_country = df_slow_wo_country.loc[df_slow_wo_country['country']!=df_slow_wo_country['location']]
print(df_slow_wo_country[['location', 'country']].value_counts().head(50))

# determine which locations are dominant for sudden-onset 
df_sudden_wo_country = df_sudden_onset[['location', 'country']].explode(['location', 'country'])
df_sudden_wo_country = df_sudden_wo_country.loc[df_sudden_wo_country['country']!=df_sudden_wo_country['location']]
print(df_sudden_wo_country[['location', 'country']].value_counts().head(50))


#%% thematic analysis

###################################################### 
#     analysis of Global North and Global South
######################################################

# create heat map
plotHeatMap(df_subcategories_filtered, 'region_overall', os.path.join(figure_path, "Disaster_all_normalized.png"))


######################################################
#   analysis of slow-onset and sudden-onset disaster  
######################################################

# create heat map
plotHeatMap(df_subcategories_filtered, 'disaster_type', os.path.join(figure_path, "Region_all_normalized.png"))



# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:59:03 2020

@author: Adam
"""
# %% Import of packages


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from datetime import timedelta
import json
from solarfun import (calculate_B_0_horizontal,
                      calculate_G_ground_horizontal,                      
                      calculate_diffuse_fraction,
                      calculate_incident_angle)
import os


# %% Model hourly global radiation on a horizontal surface in Aarhus 2018 (Martas code)


# Soruce of latitude and longitude
# https://www.gps-latitude-longitude.com/gps-coordinates-of-aarhus


# tilt representes inclination of the solar panel (in degress), orientation
# in degress (south=0)
tilt=0;
orientation=0;

lat = 56.162939 # latitude
lon = 10.203921 # longitude

year = 2018
hour_0 = datetime(year,1,1,0,0,0) - timedelta(hours=1)

hours = [datetime(year,1,1,0,0,0) 
         + timedelta(hours=i) for i in range(0,24*365)]
hours_str = [hour.strftime("%Y-%m-%d %H:%M ") for hour in hours]

timeseries = pd.DataFrame(
            index=pd.Series(
                data = hours,
                name = 'utc_time'),
            columns = pd.Series(
                data = ['B_0_h', 'K_t', 'G_ground_h', 'solar_altitude', 'F', 
                        'B_ground_h', 'D_ground_h', 'incident_angle', 
                        'B_tilted', 'D_tilted', 'R_tilted', 'G_tilted'], 
                name = 'names')
            )

# Calculate extraterrestrial irradiance
timeseries['B_0_h'] = calculate_B_0_horizontal(hours, hour_0, lon, lat)  

# Clearness index is assumed to be equal to 0.7 at every hour
timeseries['K_t']=0.7*np.ones(len(hours))  

# Calculate global horizontal irradiance on the ground
[timeseries['G_ground_h'], timeseries['solar_altitude']] = calculate_G_ground_horizontal(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate diffuse fraction
timeseries['F'] = calculate_diffuse_fraction(hours, hour_0, lon, lat, timeseries['K_t'])

# Calculate direct and diffuse irradiance on the horizontal surface
timeseries['B_ground_h']=[x*(1-y) for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]
timeseries['D_ground_h']=[x*y for x,y in zip(timeseries['G_ground_h'], timeseries['F'])]

# plot february
plt.figure(figsize=(20, 10))
gs1 = gridspec.GridSpec(2, 2)
#gs1.update(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(timeseries['G_ground_h']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='G_ground_h', color='blue')
ax1.plot(timeseries['B_ground_h']['2018-02-01 01:00':'2018-02-7 23:00'], 
         label='B_ground_h', color= 'orange')
ax1.plot(timeseries['D_ground_h']['2018-02-01 01:00':'2018-02-7 23:00'], 
         label='D_ground_h', color= 'purple')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax1.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('Horizontal surface - Global, direct and diffuse radiation of first week of february 2018')
ax2 = plt.subplot(gs1[1,0])
ax2.plot(timeseries['G_ground_h']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='G_ground_h', color='blue')
ax2.plot(timeseries['B_ground_h']['2018-06-01 01:00':'2018-06-7 23:00'], 
         label='B_ground_h', color= 'orange')
ax2.plot(timeseries['D_ground_h']['2018-06-01 01:00':'2018-06-7 23:00'], 
         label='D_ground_h', color= 'purple')
ax2.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax2.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('Horizontal surface - Global, direct and diffuse radiation of first week of june 2018')


# %% 2/3 - Diffuse fraction from measurements


# Import from csv
weather_df = pd.read_csv("weather_data.csv", sep=';',index_col='TimeStamp')  


# Drop all NaN in DF:
weather_df = weather_df.iloc[:8290]


# Calculate diffuse fraction as cloud cover/100
weather_df['F_D'] = (weather_df['Cloud']).div(100)


# Turn timestamp  into datetime
weather_df.index = pd.to_datetime(weather_df.index)


# Round to nearest hour
weather_df.index = weather_df.index.round('H')

#Insert value of midnight as the first measurement:
data = {'Temp':weather_df['Temp'][0],
        'Cloud':weather_df['Cloud'][0],
        'WindVelocity':weather_df['WindVelocity'][0],
        'WindDirection':weather_df['WindDirection'][0],
        'UV':weather_df['UV'][0],
        'F_D':weather_df['F_D'][0]} 

idx = datetime(2018, 1, 1, 0, 0)
weather_df = weather_df.append(pd.DataFrame(data, index=[idx]))
weather_df = weather_df.sort_index()

# Define timeseries
ts = pd.Series(range(len(weather_df.index)), index=weather_df.index)

# Resample series
ts = ts.resample('1H').mean()

# Convert to dataframe
df = pd.DataFrame(ts, columns=['date'])


# Merge weather_df and df into one merged where missing hours are filled with NaN.
df_merged = df.merge(weather_df, how='outer', left_index=True, right_index=True)

## Data is somehow duplicated some places...duplicates are removed but doesnt really have a effect.
df_merged = df_merged[~df_merged.index.duplicated(keep='first')]

# Interpolate between missing values of F_D
df_merged['F_D'] = df_merged['F_D'].interpolate(limit_area='inside')



# Calculate direct and diffuse irradiance on the horizontal surface in Aarhus
df_merged['B_ground_h']=[x*(1-y) for x,y in zip(timeseries['G_ground_h'], df_merged['F_D'])]
df_merged['D_ground_h']=[x*y for x,y in zip(timeseries['G_ground_h'], df_merged['F_D'])]





# plot february
plt.figure(figsize=(20, 10))
gs1 = gridspec.GridSpec(2, 2)
#gs1.update(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(df_merged['B_ground_h']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='B_ground_h', color='orange')
ax1.plot(df_merged['D_ground_h']['2018-02-01 01:00':'2018-02-7 23:00'], 
         label='D_ground_h', color= 'purple')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax1.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('Horizontal surface - Direct and diffuse radiation of first week of february 2018')
ax2 = plt.subplot(gs1[1,0])
ax2.plot(df_merged['B_ground_h']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='B_ground_h', color='orange')
ax2.plot(df_merged['D_ground_h']['2018-06-01 01:00':'2018-06-7 23:00'], 
         label='D_ground_h', color= 'purple')
ax2.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax2.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('Horizontal surface - Direct and diffuse radiation of first week of june 2018')


#%% 4/5 Rooftop PV installations


# 13,5 degrees tilt and 45 degrees souteast


# tilt representes inclination of the solar panel (in degress), orientation
# in degress (south=0)
tilt=13.5;
orientation=45;

lat = 56.162939 # latitude
lon = 10.203921 # longitude


# Clearness index is assumed to be equal to 0.7 at every hour
df_merged['K_t']=0.7*np.ones(len(df_merged.index))  

# Calculate incident angle
df_merged['incident_angle'] = calculate_incident_angle(df_merged.index[0:], hour_0, lon, lat,  tilt, orientation)

# Calculate global horizontal irradiance on the ground and solar altitude
[df_merged['G_ground_h'], df_merged['solar_altitude']] = calculate_G_ground_horizontal(df_merged.index[0:], hour_0, lon, lat, df_merged['K_t'])


# Calculate extraterrestrial irradiance
df_merged['B_0_h'] = calculate_B_0_horizontal(df_merged.index[0:], hour_0, lon, lat)

# Initialize
df_merged['B_tilt']= np.ones(len(df_merged.index))
df_merged['D_tilt']= np.ones(len(df_merged.index))
df_merged['R_tilt']= np.ones(len(df_merged.index))

# Calculate  B_tilted
for i in range(0,len(df_merged.index)):
    df_merged['B_tilt'][i] = (df_merged['B_ground_h'][i]*max([0, np.cos(df_merged['incident_angle'][i]*np.pi/180)]))/(np.sin(df_merged['solar_altitude'][i]*np.pi/180))
    if df_merged['B_tilt'][i]>0:
        df_merged['B_tilt'][i] = df_merged['B_tilt'][i]
    else:
        df_merged['B_tilt'][i]=0
        
# Calculate diffuse radiance assuming circumsolar:
#df_merged['D_tilt']=df_merged['D_ground_h']*((1+np.cos(tilt*np.pi/180))/1)


df_merged['k1'] = (df_merged['B_ground_h']/df_merged['B_0_h']).fillna(0)

for i in range(0,len(df_merged.index)):
    df_merged['D_tilt'][i] = df_merged['k1'][i]*(df_merged['D_ground_h'][i]/(np.sin(df_merged['solar_altitude'][i]*np.pi/180)))*max([0, np.cos(df_merged['incident_angle'][i]*np.pi/180)])
    if df_merged['D_tilt'][i]>0:
        df_merged['D_tilt'][i] = df_merged['D_tilt'][i]
    else:
        df_merged['D_tilt'][i]=0


# Reflectivity index:
df_merged['rho'] = 0.05*np.ones(len(df_merged.index))

# Calculate albedo irradiance:
df_merged['R_tilt'] = df_merged['rho']*df_merged['G_ground_h']*((1-np.cos(tilt*np.pi/180))/2)

for i in range(0,len(df_merged.index)):
    df_merged['R_tilt'][i] = df_merged['rho'][i]*df_merged['G_ground_h'][i]*((1-np.cos(tilt*np.pi/180))/2)
    if df_merged['R_tilt'][i]>0:
        df_merged['R_tilt'][i] = df_merged['R_tilt'][i]
    else:
        df_merged['R_tilt'][i]=0



# Sum up to obtain global radiation
df_merged['G_tilt'] = df_merged['B_tilt'] + df_merged['D_tilt'] + df_merged['R_tilt']



# plot february
plt.figure(figsize=(20, 10))
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(df_merged['B_tilt']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='B_tilt', color='orange')
ax1.plot(df_merged['D_tilt']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='D_tilt', color='purple')
ax1.plot(df_merged['R_tilt']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='R_tilt', color='darkgreen')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax1.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('PV-panels of Navitas - Direct, diffuse and albedo radiation - first week of february 2018')
ax2 = plt.subplot(gs1[1,0])
ax2.plot(df_merged['B_tilt']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='B_tilt', color='orange')
ax2.plot(df_merged['D_tilt']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='D_tilt', color='purple')
ax2.plot(df_merged['R_tilt']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='R_tilt', color='darkgreen')
ax2.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax2.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('PV-panels of Navitas - Direct, diffuse and albedo radiation - first week of june 2018')



# plot february
plt.figure(figsize=(20, 10))
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(df_merged['G_tilt']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='G_tilt', color='blue')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax1.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('PV-panels of Navitas - Global radiation of first week of february 2018')
ax2 = plt.subplot(gs1[1,0])
ax2.plot(df_merged['G_tilt']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='G_tilt', color='blue')
ax2.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax2.set_ylabel('$W/m^2$',fontsize=20, color="black")
plt.title('PV-panels of Navitas - Global radiation of first week of june 2018')


#%% 6 Power of installations


# Interpolate between missing values of temperature
df_merged['Temp'] = df_merged['Temp'].interpolate(limit_area='inside')

# Pmax temperature coefficient
gamma_T = -0.0044

# Area of one cell:
A_cell = 0.156*0.156 #m^2

# Total area pr. panel:
A_panel = 6*10*A_cell #m^2

# Total area for installation
A_total = A_panel*1000 #m^2

# Temperature dependent efficiency
eta_STC = ((16+15.7)/2)*1e-2    #Effeciency at Standard Test Conditions
T_cell_STC = 25                 #Temperature of the cell under Standard Test Conditions
G_NOCT = 800                  #Nominel Operating Cell Temperature
T_amb_NOCT = 20                 #Ambient tempeature at Nominel Operating Cell Temperature
T_cell_NOCT = 45                #Cell temperature under Nominiel Operating Cell Temperature
#eta_system = 0.9               #System efficiency
P_STC = 255                     #Watt

G_STC = 1000 # radiation under standard test conditions

# Initialize
df_merged['T_cell']= np.ones(len(df_merged.index))
df_merged['eta']= np.ones(len(df_merged.index))
df_merged['Generated Power - Calculated'] = np.ones(len(df_merged.index))


for i in range(0,len(df_merged.index)):
    if df_merged['G_tilt'][i]>0:
        df_merged['T_cell'][i] =  df_merged['Temp'][i] + (T_cell_NOCT - T_amb_NOCT)*(np.divide(df_merged['G_tilt'][i],G_NOCT))
    else:
        df_merged['T_cell'][i]=df_merged['Temp'][i]

for i in range(0,len(df_merged.index)):
    df_merged['Generated Power - Calculated'][i] = (np.divide(df_merged['G_tilt'][i],G_STC))*(1+gamma_T*(df_merged['T_cell'][i]-T_cell_STC))*(P_STC)
    if df_merged['Generated Power - Calculated'][i] < 0:
        df_merged['Generated Power - Calculated'][i]=0
    else:
        df_merged['Generated Power - Calculated'][i]=df_merged['Generated Power - Calculated'][i]

# %% 7 Measured power

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

#df_import_power = pd.read_excel('CTS Data Aflæsning Strom2.xlsx', usecols='B,G:AD', header=2, index_col=0)
df_import_power = pd.read_excel('CTS Data Aflæsning Strom.xlsx', usecols='B,G:AD', header=2, index_col=0)
df_import_power.index = df_import_power.index.date

# Initialize
df_merged['Generated Power - Measured']= np.ones(len(df_merged.index))
np_import_power = df_import_power.to_numpy()
#array_import_power = np.zeros(len(np_import_power))
#array_import_power = np.zeros(31*24-1)
array_import_power = df_merged['Generated Power - Calculated'][0:31*24-1]
#array_import_power = np_import_power[0,:]
for i in range(0,len(np_import_power)):
    #array_import_power = np.append(imp_p,imp_P[i,:])
    array_import_power = np.append(array_import_power,np_import_power[i,:])
    
#df_merged['Generated Power - Measured'][0] = 0
for i in range(0,len(df_merged.index)):
    df_merged['Generated Power - Measured'][i] = array_import_power[i]



# %% Plot of the power produced and measured
# plot february
plt.figure(figsize=(20, 10))
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(gs1[0,0])
ax1.plot(df_merged['Generated Power - Calculated']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='Generated Power - Calculated', color='orange')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax1.plot(df_merged['Generated Power - Measured']['2018-02-1 01:00':'2018-02-7 23:00'], 
         label='Generated Power - Measured', color='brown')
ax1.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax1.set_ylabel('$kW$',fontsize=20, color="black")
plt.title('Calculated Power Produced by PV installation of first week of february 2018')


# Plot June
ax2 = plt.subplot(gs1[1,0])
ax2.plot(df_merged['Generated Power - Calculated']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='Generated Power - Calculated', color='orange')
ax2.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax2.plot(df_merged['Generated Power - Measured']['2018-06-1 01:00':'2018-06-7 23:00'], 
         label='Generated Power - Measured', color='brown')
ax2.legend(fancybox=True, shadow=True,fontsize=12, loc='upper right')
ax2.set_ylabel('$kW$',fontsize=20, color="black")
plt.title('Calculated Power Produced by PV installation of first week of june 2018')


# %% Root Mean Square - Daily

# # Hourly
# hourly_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='h')).mean() 
# hourly_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='h')).mean() 
# df_merged['diff_hourly_squared'] = [(x-y)**2 for x,y in zip(hourly_mean_P_cal, hourly_mean_P_meas)]
# RMSE_hourly =  np.sqrt(np.sum( df_merged['diff_hourly_squared'] ) / len(df_merged.index) )
# df_merged['diff_hourly'] = [x-y for x,y in zip(hourly_mean_P_cal, hourly_mean_P_meas)]
# RRMSE_hourly = RMSE_hourly / np.mean(df_merged['diff_hourly'])
# print('RMSE Hourly  = ',RMSE_hourly)
# print('   Relative  = ',RRMSE_hourly)

# # Daily
# daily_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='d')).mean() 
# daily_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='d')).mean() 
# diff_daily_squared = [(x-y)**2 for x,y in zip(daily_mean_P_cal, daily_mean_P_meas)]
# RMSE_daily =  np.sqrt(np.sum( diff_daily_squared ) / 365 )
# diff_daily = [x-y for x,y in zip(daily_mean_P_cal, daily_mean_P_meas)]
# RRMSE_daily = RMSE_daily / np.mean(diff_daily)
# print('RMSE Daily   = ',RMSE_daily)
# print('  Relative   = ',RRMSE_daily)

# # Weekly 
# weekly_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='w')).mean() 
# weekly_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='w')).mean() 
# diff_weekly_squared = [(x-y)**2 for x,y in zip(weekly_mean_P_cal, weekly_mean_P_meas)]
# RMSE_weekly =  np.sqrt(np.sum( diff_weekly_squared ) / 52 )
# diff_weekly = [x-y for x,y in zip(weekly_mean_P_cal, weekly_mean_P_meas)]
# RRMSE_weekly = RMSE_weekly / np.mean(diff_weekly)
# print('RMSE Weekly  = ',RMSE_weekly)
# print('   Relative  = ',RRMSE_weekly)

# # Monthly
# monthly_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='M')).mean() 
# monthly_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='M')).mean() 
# diff_monthly_squared = [(x-y)**2 for x,y in zip(monthly_mean_P_cal, monthly_mean_P_meas)]
# RMSE_monthly =  np.sqrt(np.sum( diff_monthly_squared ) / 12 )
# diff_monthly = [x-y for x,y in zip(monthly_mean_P_cal, monthly_mean_P_meas)]
# RRMSE_monthly = RMSE_monthly / np.mean(diff_monthly)
# print('RMSE Monthly = ',RMSE_monthly)
# print('    Relative = ',RRMSE_monthly)

# %% Root Mean Square - Hourly

#Initialize
df_merged['RMSE_hourly']= np.ones(len(df_merged.index))
df_merged['RRMSE_hourly']= np.ones(len(df_merged.index))
RMSE_daily= np.array(np.ones(365))
RRMSE_daily= np.array(np.ones(365))
RMSE_weekly= np.array(np.ones(52))
RRMSE_weekly= np.array(np.ones(52))
RMSE_monthly= np.array(np.ones(12))
RRMSE_monthly= np.array(np.ones(12))

# Hourly
for i in range(0,len(df_merged.index)-743):
    df_merged['RMSE_hourly'][i] = np.sqrt(((df_merged['Generated Power - Calculated'][i+743]-df_merged['Generated Power - Measured'][i+743])**2)/1)
    df_merged['RRMSE_hourly'][i] = df_merged['RMSE_hourly'][i] / ((df_merged['Generated Power - Calculated'][i+743]+df_merged['Generated Power - Measured'][i+743])/2)
    if (df_merged['Generated Power - Calculated'][i+743]+df_merged['Generated Power - Measured'][i+743]) ==0:
        df_merged['RRMSE_hourly'][i] = 0
        
mean_RRMSE_hourly = np.mean(df_merged['RRMSE_hourly'])
print('  RRMSE Hourly  = ', format(mean_RRMSE_hourly*1e2,'.2f'),'%')
mean_RMSE_hourly = np.mean(df_merged['RMSE_hourly'])

# Daily
daily_power_calculated = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='d')).mean() 
daily_power_measured = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='d')).mean() 
for i in range(0,365-31):
    RMSE_daily[i] = np.sqrt(((daily_power_calculated[i+31]-daily_power_measured[i+31])**2)/24)
    RRMSE_daily[i] = RMSE_daily[i] / ((daily_power_calculated[i+31]+daily_power_measured[i+31])/2)
    if (daily_power_calculated[i+31]+daily_power_measured[i+31]) ==0:
        RRMSE_daily[i] = 0
        
mean_RRMSE_daily = np.mean(RRMSE_daily)
print('  RRMSE Daily   = ', format(mean_RRMSE_daily*1e2,'.2f'),'%')
mean_RMSE_daily = np.mean(RMSE_daily)

# Weekly
weekly_power_calculated = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='w')).mean() 
weekly_power_measured = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='w')).mean() 
for i in range(0,52-4):
    RMSE_weekly[i] = np.sqrt(((weekly_power_calculated[i+4]-weekly_power_measured[i+4])**2)/168)
    RRMSE_weekly[i] = RMSE_weekly[i] / ((weekly_power_calculated[i+4]+weekly_power_measured[i+4])/2)
    if (weekly_power_calculated[i+4]+weekly_power_measured[i+4]) ==0:
        RRMSE_weekly[i] = 0
        
mean_RRMSE_weekly = np.mean(RRMSE_weekly)
print('  RRMSE Weekly  = ', format(mean_RRMSE_weekly*1e2,'.2f'),'%')
mean_RMSE_weekly = np.mean(RMSE_weekly)

# Monthly
monthly_power_calculated = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='M')).mean() 
monthly_power_measured = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='M')).mean() 
for i in range(0,11):
    RMSE_monthly[i] = np.sqrt(((monthly_power_calculated[i+1]-monthly_power_measured[i+1])**2)/672)
    RRMSE_monthly[i] = RMSE_monthly[i] / ((monthly_power_calculated[i+1]+monthly_power_measured[i+1])/2)
    if (monthly_power_calculated[i+1]+monthly_power_measured[i+1]) ==0:
        RRMSE_monthly[i] = 0
        
mean_RRMSE_monthly = np.mean(RRMSE_monthly)
print('  RRMSE Monthly = ', format(mean_RRMSE_monthly*1e2,'.2f'),'%')
mean_RMSE_monthly = np.mean(RMSE_monthly)


# # %% Root Mean Square
# hourly_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='h')).mean() 
# hourly_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='h')).mean() 
# df_merged['diff_hourly_squared'] = [(x-y)**2 for x,y in zip(df_merged['Generated Power - Calculated'], df_merged['Generated Power - Measured'])]
# df_merged['RMSE_hourly'] =  np.mean([(x/1)**(1/2) for x in zip(df_merged['diff_hourly_squared'])])
# df_merged['diff_hourly'] = [x-y for x,y in zip(hourly_mean_P_cal, hourly_mean_P_meas)]
# df_merged['RRMSE_hourly'] = np.mean(df_merged['RMSE_hourly'] / np.mean(df_merged['diff_hourly']))
# print('RMSE Hourly  = ',df_merged['RMSE_hourly'])
# print('   Relative  = ',df_merged['RRMSE_hourly'])

# # Daily
# daily_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='d')).mean() 
# daily_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='d')).mean() 
# diff_daily_squared = [(x-y)**2 for x,y in zip(daily_mean_P_cal, daily_mean_P_meas)]
# RMSE_daily =  np.sqrt(np.sum( diff_daily_squared ) / 365 )
# diff_daily = [x-y for x,y in zip(daily_mean_P_cal, daily_mean_P_meas)]
# RRMSE_daily = RMSE_daily / np.mean(diff_daily)
# print('RMSE Daily   = ',RMSE_daily)
# print('  Relative   = ',RRMSE_daily)

# # Weekly 
# weekly_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='w')).mean() 
# weekly_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='w')).mean() 
# diff_weekly_squared = [(x-y)**2 for x,y in zip(weekly_mean_P_cal, weekly_mean_P_meas)]
# RMSE_weekly =  np.sqrt(np.sum( diff_weekly_squared ) / 52 )
# diff_weekly = [x-y for x,y in zip(weekly_mean_P_cal, weekly_mean_P_meas)]
# RRMSE_weekly = RMSE_weekly / np.mean(diff_weekly)
# print('RMSE Weekly  = ',RMSE_weekly)
# print('   Relative  = ',RRMSE_weekly)

# # Monthly
# monthly_mean_P_cal = df_merged['Generated Power - Calculated'].groupby(pd.Grouper(freq='M')).mean() 
# monthly_mean_P_meas = df_merged['Generated Power - Measured'].groupby(pd.Grouper(freq='M')).mean() 
# diff_monthly_squared = [(x-y)**2 for x,y in zip(monthly_mean_P_cal, monthly_mean_P_meas)]
# RMSE_monthly =  np.sqrt(np.sum( diff_monthly_squared ) / 12 )
# diff_monthly = [x-y for x,y in zip(monthly_mean_P_cal, monthly_mean_P_meas)]
# RRMSE_monthly = RMSE_monthly / np.mean(diff_monthly)
# print('RMSE Monthly = ',RMSE_monthly)
# print('    Relative = ',RRMSE_monthly)

# %%

# Export to Emil

df_merged.to_csv(r'C:\Users\Adam\OneDrive - Aarhus universitet\Solar Energy Project\PROJECT 2\export_dataframe.csv', header=True)





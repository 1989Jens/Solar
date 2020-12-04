# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 12:28:51 2018

@author: Marta Victoria

Solar functions to calculate radiation on the horizonal ground and on tilted 
surfaces

"""

import numpy as np

def eccentricity(day): 
    """
    Calculate eccentricity.
    
    Parameters:
        day = number of the day, counted from the first day of the year 
              (1...365)    
    """
    ecc = 1 + 0.033*np.cos(360/365*day*np.pi/180)
    return ecc
     

def declination(day): 
    """
    Calculate declination, in degrees.
    
    Parameters:
        day = number of the day, counted from the first day of the year 
              (1...365)   
    """    
    declination = 23.45*np.sin(360/365*(day+284)*np.pi/180) 
    return declination


#def declination(day): # alternative definition
#    """
#    Calculate declination.
#    Parameters:
#        day = number of the day, counted from the first day of the year 
#              (1...365)    
#    """
#    dayAngle = day*2*np.pi/365 
#    declination = 180/np.pi*(0.006918  -  0.399912*np.cos(dayAngle)
#               + 0.070257*np.sin(dayAngle) 
#				  -  0.006758*np.cos(2*dayAngle) + 0.000907*np.sin(2*dayAngle) 
#				  -  0.002697*np.cos(3*dayAngle) + 0.00148*np.sin(3*dayAngle))
#    return declination

    
def solar_altitude(latitude, declination, omega): 
    """
    Calculate solar altitude, in degrees
    
    Parameters:
        latitude = in degrees
        declination = in degrees
        omega = true solar time, expressed as angle (omega=0 when the Sun is a 
                the highest position), in degrees    
    """    
    solar_altitude = (180/np.pi)*(np.arcsin(np.sin(declination*np.pi/180)
                     *np.sin(latitude*np.pi/180) 
                     + (np.cos(declination*np.pi/180)*np.cos(latitude*np.pi/180)
                     *np.cos(omega*np.pi/180))))
    return solar_altitude


def solar_azimuth (latitude, declination, omega): 
    """
    Calculate solar azimuth, in degrees
    
    Parameters:
        latitude = in degrees
        declination = in degrees
        omega = true solar time, expressed as angle (omega=0 when the Sun is a 
                the highest position), in degrees 
    
    """
    gamma_s=solar_altitude(latitude, declination, omega)
    if latitude>0:
        sign=1
    else:
        sign=-1
    solar_azimuth = (180/np.pi)*(np.arccos(sign*(np.sin(gamma_s*np.pi/180)
                    *np.sin(latitude*np.pi/180)-np.sin(declination*np.pi/180))
                    /(np.cos(gamma_s*np.pi/180)*np.cos(latitude*np.pi/180)))) 
    return solar_azimuth    


#def solar_azimuth (latitude, declination, omega): # alternative definition
#    """
#    Calculate solar azimuth.
#    Parameters:
#        latitude = in degrees
#        declination = in degrees
#        omega = true solar time, expressed as angle (omega=0 when the Sun is a 
#                the highest position), in degrees 
#    
#    """
#    gamma_s=solar_altitude(latitude, declination, omega)
#    solar_azimuth = (180/np.pi)*(np.arcsin(np.cos(declination*np.pi/180)
#                    *np.sin(omega*np.pi/180)/np.cos(gamma_s*np.pi/180)))
#    
#    if np.cos(declination*np.pi/180) < np.tan(declination*np.pi/180)/np.tan(latitude*np.pi/180):
#        if omega < 0:
#            solar_azimuth = -180 + np.abs(solar_azimuth)
#        else:
#            solar_azimuth = 180 - solar_azimuth
#
#    return solar_azimuth

    
def ET(day):
    """
    Calculate correction for different lengths of the days in a year
    
    Parameters:
        day = number of the day, counted from the first day of the year (1...365)
    """
    B = (day-81)*2*np.pi/364
    ET_ = 9.87*np.sin(2*B) - 7.53*np.cos(B)-1.5*np.sin(B)
    return ET_


def omega(hour, day, longitude): 
    """
    Calculate true solar time, expressed as angle (omega=0 when the Sun is a 
    the highest position), in degrees
    
    Parameters:
        hour 
        day = number of the day, counted from the first day of the year (1...365)
        longitude = in degrees
    
    """
    # TO- AO = UCT (hour is expressed in UCT)
    # reference longitude = 0 (Greenwich)

    ET_=ET(day)
    omega = 15*(hour + ET_/60 - 12) + (longitude)
    return omega
    

#def airmass(solar_altitude): 
#    """
#    Calculate airmass
#    Parameters:
#        solar_altitude = altitude, in degrees
#        
#    """
#    if (np.sin(solar_altitude*np.pi/180)) < 0.001:
#        airmass = 1/0.001
#    else:
#        airmass = 1/(np.sin(solar_altitude*np.pi/180))
#    return airmass


def B_0_horizontal(day,solar_altitude): 
    """
    Calculate direct irradiance on the horizontal ground surface, in W/m2
    
    Parameters:
        day = number of the day, counted from the first day of the year (1...365)    
        solar _altitude = in degrees
    """    
    B_0 = 1367 # [W/m2]
    B_0_horizontal=B_0*eccentricity(day)*np.sin(solar_altitude*np.pi/180)
    return B_0_horizontal


def calculate_B_0_horizontal(hours_year, hour_0, longitude, latitude):
    """
    Calculate direct irradiance on the horizontal ground surface, in W/m2
    for a series of hours 
    
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
    """ 
    solar_altitude_ = [solar_altitude(latitude, declination((hour-hour_0).days), 
                      omega(hour.hour, (hour-hour_0).days, longitude)) 
                      for hour in hours_year]
    
 
    solar_altitude_ = [x if x>0 else 0 for x in solar_altitude_]   
    B_0_horizontal_ = [B_0_horizontal((hour-hour_0).days, solar_altitude) 
                            for hour, solar_altitude 
                            in zip(hours_year,solar_altitude_)]
    return B_0_horizontal_


#def G_ground_horizontal(day,solar_altitude): # alternative definition, if you 
#    """                                      # don't have information on
                                              # the clearness index
#    Calculate global irradiance on the horizontal ground surface, in W/m2
#    Parameters:
#        day = number of the day, counted from the first day of the year (1...365)    
#        solar _altitude = in degrees
#    """
#    
#    if np.sin(solar_altitude*np.pi/180) < np.sin(1*np.pi/180):
#        G_ground_horizontal=0
#    else:
#        B_0= 1367 # [W/m2]
#        G_ground_horizontal=B_0*(0.74**((1/(np.sin(solar_altitude*np.pi/180)))**0.678))*np.sin(solar_altitude*np.pi/180)
#        #*eccentricity(day)
#        #it can be multiplied by included 10% diffuse radiation
#    return G_ground_horizontal

def G_ground_horizontal(day, solar_altitude, clearness_index):
    """
    Calculate global irradiance on the horizontal ground surface, in W/m2

    Parameters:
        day = number of the day, counted from the first day of the year (1...365)    
        solar _altitude = in degrees
        clearness_index
    """
    G_ground_horizontal = clearness_index*B_0_horizontal(day,solar_altitude)    
    return G_ground_horizontal

    
def calculate_G_ground_horizontal(hours_year, hour_0, longitude, latitude, clearness_index_):
    """
    Calculate global irradiance on the horizontal ground surface, in W/m2
    for a series of hours 
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
        clearnes_index_ = list of clearness indices for the list of hours
    """ 
    solar_altitude_ = [solar_altitude(latitude, declination((hour-hour_0).days), 
                       omega(hour.hour, (hour-hour_0).days, longitude)) 
                       for hour in hours_year]
    solar_altitude_ = [x if x>0 else 0. for x in solar_altitude_]

    G_ground_horizontal_ = [G_ground_horizontal((hour-hour_0).days, 
                            solar_altitude, clearness_index) 
                            for hour, solar_altitude, clearness_index 
                            in zip(hours_year,solar_altitude_, clearness_index_)]
    return G_ground_horizontal_ , solar_altitude_ 

    
def diffuse_fraction(solar_altitude, clearness_index):
    """
    Calculate diffuse fraction
    
    Parameteres:
        K_t = Clearnes index    
        solar _altitude = in degrees
    """
    K_t=clearness_index
    
    if K_t <= 0.3:
        diffuse_fraction_ = np.min([1, (1.02 - 0.254*K_t + 0.0123*np.sin(solar_altitude*np.pi/180))])
        
    elif (K_t > 0.3 and K_t <= 0.78):
        diffuse_fraction_ = np.min([0.97, np.max([0.1, 1.4-1.749*K_t+0.177*np.sin(solar_altitude*np.pi/180)])])
    else:
        diffuse_fraction_ =np.max([0.1, 0.486*K_t-0.182*np.sin(solar_altitude*np.pi/180)])
        
    return diffuse_fraction_


def calculate_diffuse_fraction(hours_year, hour_0, longitude, latitude, clearness_index_):
    """
    Calculate difussion fraction for a series of hours 
   
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
        clearnes_index_ = list of clearness indices for the list of hours
    """ 
    solar_altitude_ = [solar_altitude(latitude, declination((hour-hour_0).days), 
                       omega(hour.hour, (hour-hour_0).days, longitude)) 
                       for hour in hours_year]
    difuse_fraction_ = [diffuse_fraction(solar_altitude, clearness_index) 
                            for solar_altitude, clearness_index 
                            in zip(solar_altitude_, clearness_index_)]
    return difuse_fraction_
    

def incident_angle(hour, hour_0, longitude, latitude, tilt, orientation):    
    """
    Calculate the angle that forms the radio-vector of the Sun and the normal 
    of the surface
    
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
        tilt = tilting angle of the surface, in degrees
        orientation = orientation of the surface (south=0), in degrees
    """ 
    if latitude>=0:
        sign=1
    else:
        sign=-1
    #aproximatted value when orientaton is south
#    incident_angle_ = ((180/np.pi)*np.arccos(sign*np.sin(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*(np.abs(latitude)-tilt)) 
#    + np.cos(np.pi/180*declination((hour-hour_0).days))*np.cos(np.pi/180*(np.abs(latitude)-tilt))*np.cos(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))))

    cos_incident_angle = (np.sin(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*latitude)*np.cos(np.pi/180*tilt)
                        - sign*np.sin(np.pi/180*declination((hour-hour_0).days))*np.cos(np.pi/180*latitude)*np.sin(np.pi/180*tilt)*np.cos(np.pi/180*orientation)
                        + np.cos(np.pi/180*declination((hour-hour_0).days))*np.cos(np.pi/180*latitude)*np.cos(np.pi/180*tilt)*np.cos(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))
                        + sign*np.cos(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*latitude)*np.sin(np.pi/180*tilt)*np.cos(np.pi/180*orientation)*np.cos(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))
                        + np.cos(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*orientation)*np.sin(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))*np.sin(np.pi/180*tilt))
    #The expresion above could be >1 due to precison, this will raise an error when calculating the arccos(cos_incident angle)
    if cos_incident_angle > 1. :
        cos_incident_angle=1
    incident_angle_ = (180/np.pi)*np.arccos(cos_incident_angle)                    

    return incident_angle_

def calculate_incident_angle(hours_year, hour_0, longitude, latitude,  tilt, orientation):
    """
    Calculate the angle that forms the radio-vector of the Sun and the normal 
    of the surface for a series of hours
    
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
        tilt = tilting angle of the surface, in degrees
        orientation = orientation of the surface (south=0), in degrees
    """ 
    
    incident_angle_=[incident_angle(hour, hour_0, longitude, latitude, tilt, orientation) for hour in hours_year]
    return incident_angle_


def incident_angle_haxis(hour, hour_0, longitude, latitude,  tilt, orientation):
    """
    Calculate the incident angle for horizontal-axis tracking
    
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
        tilt = tilting angle of the surface, in degrees
        orientation = orientation of the surface (south=0), in degrees
    """ 
    if latitude>0:
        sign=1
    else:
        sign=-1
    betaNS=0
    beta_trian=betaNS-np.abs(latitude)
    tan_phiNS=np.sin(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))/(np.cos(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))*np.cos(np.pi/180*beta_trian)-sign*np.tan(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*beta_trian))
    phiNS=180/np.pi*np.arctan(tan_phiNS)
    cos_incident_angle_haxis = np.cos(phiNS*np.pi/180)*(np.cos(np.pi/180*declination((hour-hour_0).days))*np.cos(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))*np.cos(np.pi/180*beta_trian)
                               -sign*np.sin(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*beta_trian)
                               +np.sin(phiNS*np.pi/180)*np.cos(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude)))                       
    if cos_incident_angle_haxis > 1. :
        cos_incident_angle_haxis=1
    incident_angle_haxis_ = (180/np.pi)*np.arccos(cos_incident_angle_haxis)  
    return incident_angle_haxis_


def tilt_angle_haxis(hour, hour_0, longitude, latitude): 
    """
    Calculate the tilt angle for horizontal-axis tracking
    
    Parameters:
        hours_year = series of hours for which B_0_horizontal is calculated
        hours_0 = reference hour
        longitude = in degrees    
        latitude = in degrees
    """ 
    tan_tilt_angle_haxis = np.sin(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))/(np.cos(np.pi/180*omega(hour.hour, (hour-hour_0).days, longitude))*np.cos(np.pi/180*latitude)+ np.tan(np.pi/180*declination((hour-hour_0).days))*np.sin(np.pi/180*latitude))              
    tilt_angle_haxis_ = (180/np.pi)*np.arctan(tan_tilt_angle_haxis)  
    return tilt_angle_haxis_



def Gaussian_tilt_orientation(inclination_mean=0, inclination_sd=0, azimuth_mean=0, azimuth_sd=0):
    """
    Calculate the weights, inclination, and orientation based on Gaussian 
    distributions of the inclinations and orientations of panels.
    REatlas considers azimuth = 0 as south oriented panel
    """ 
    from scipy.stats import norm
    
    if azimuth_sd==0:
        x_azimuth=np.array([azimuth_mean])
        prob_azimuth=np.array([1])
    else:
        #min inclination limit to 0º, max inclination limit to 90º
        x_azimuth = np.arange(np.max([-90,azimuth_mean-2*azimuth_sd]),np.min([90,(azimuth_mean+3*azimuth_sd)]),azimuth_sd)
        #x_azimuth = np.arange(np.max([-90,azimuth_mean-2*azimuth_sd]),np.min([90,(azimuth_mean+3*azimuth_sd)]),azimuth_sd/2)
        prob_azimuth = norm.pdf(x_azimuth,azimuth_mean,azimuth_sd)
    
    if inclination_sd==0:
        x_inclination = np.array([inclination_mean])
        prob_inclination=np.array([1])
    else:
        #min inclination limit to 0º, max inclination limit to 90º
        x_inclination = np.arange(np.max([0,inclination_mean-2*inclination_sd]), np.min([90,(inclination_mean+3*inclination_sd)]),inclination_sd)
        #x_inclination = np.arange(np.max([0,inclination_mean-2*inclination_sd]), np.min([90,(inclination_mean+3*inclination_sd)]),inclination_sd/2)
        prob_inclination = norm.pdf(x_inclination,inclination_mean,inclination_sd)

    azimuths = []
    inclinations = []
    weights = []
    for i,x in enumerate(x_azimuth):
        for j,y in enumerate(x_inclination):
            azimuths.append(x)
            inclinations.append(y)
            weights.append(prob_azimuth[i]*prob_inclination[j])
    weights=weights/sum(weights)

    #plot
    #plt.figure(figsize=(15, 5))
    #gs1 = gridspec.GridSpec(1, 2)
    #ax1 = plt.subplot(gs1[0,0])
    #ax1.set_xlabel('azimuth ($\circ$)', fontsize=16)
    #ax1.set_ylabel('prob azimuth', fontsize=16)
    #plt.xticks(fontsize=16)
    #plt.yticks([])
    #ax1.plot(x_azimuth, prob_azimuth)
    #ax1.xaxis.grid(True)
    #ax2 = plt.subplot(gs1[0,1])
    #ax2.set_xlabel('inclination ($\circ$)', fontsize=16)
    #ax2.set_ylabel('prob inclination', fontsize=16)
    #plt.xticks(fontsize=16)
    #plt.yticks([])
    #ax2.plot(x_inclination, prob_inclination)
    #ax2.xaxis.grid(True)
    return weights, inclinations, azimuths


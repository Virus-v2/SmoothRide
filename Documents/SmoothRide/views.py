from flask import render_template
from flask import request
from flaskexample import app
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.models import model_from_json
from .models import *

# Indicate when modules have been loaded
print('Modules loaded!')

@app.route('/')
def smooth_input():
    return render_template("input.html")

@app.route('/slides')
def smooth_input():
    return render_template("slides.html")

@app.route('/output')
def smooth_output():
    #Retrieve origin and destination information
    address_start= request.args.get('user_address_start')
    address_end= request.args.get('user_address_end')

    #Find directions
    t = directions(address_start, address_end)

    #Join polylines and retrieve lat and long
    polyline_list1,route_dist1 = find_points_and_join(t,0)

    #Change the format of the lat and long
    encode_list1 = rearrange_polyline(polyline_list1)

    #Convert from lat and long to UTM coordinates
    coordinate1 = UTM_coord(encode_list1)

    # Make a Shapely linestring
    line1 = make_linestring(coordinate1)

    # Interpolate polyline at regular intervals
    interp_output1 = interpolate_polyline(line1)

    # Make a string for each interval
    poly_strings1 = make_poly_strings(interp_output1[0])

    # Download Streetview Images
    streetview_output1 = retrieve_streetview(interp_output1[1])

    # Make plots and map
    result1 = make_graphics(poly_strings1,0,0)

    # Join the javascript code
    map_code1 = ''.join(result1[1])

    # Return key points
    center_point1 = result1[2]
    start_point =result1[4]
    end_point = result1[5]

    # Return key metrics
    dist_1 = result1[6]
    el_range1 = result1[7]
    red_over_green1 = result1[8]
    i_in = result1[3] - 1

    #Join polylines and retrieve lat and long
    polyline_list2, route_dist2 = find_points_and_join(t,1)

    #Change the format of the lat and long
    encode_list2 = rearrange_polyline(polyline_list2)


    #Convert from lat and long to UTM coordinates
    coordinate2 = UTM_coord(encode_list2)

    # Make a Shapely linestring
    line2 = make_linestring(coordinate2)

    # Interpolate polyline at regular intervals
    interp_output2 = interpolate_polyline(line2)

    # Make a string for each interval
    poly_strings2 = make_poly_strings(interp_output2[0])

    # Download Streetview Images
    streetview_output2 = retrieve_streetview(interp_output2[1])

    # Make plots and map
    result2 = make_graphics(poly_strings2,i_in,1)

    # Join the javascript code
    map_code2 = ''.join(result2[1])

    # Return key metrics
    dist_2 = result2[6]
    el_range2 = result2[7]
    red_over_green2 = result2[8]

    # Determine which route  meets the given condition
    if route_dist1 > route_dist2 and el_range1 > el_range2 and red_over_green1 > red_over_green2:
        condition = "Route 2 is the shortest, flattest, and smoothest."
    elif route_dist1 > route_dist2 and el_range1 < el_range2 and red_over_green1 > red_over_green2:
         condition = "Route 1 is the flattest, and Route 2 is the shortest and smoothest."
    elif route_dist1 > route_dist2 and el_range1 < el_range2 and red_over_green1 < red_over_green2:
         condition = "Route 1 is the smoothest and flattest, and Route 2 is the shortest."
    elif route_dist1 < route_dist2 and el_range1 < el_range2 and red_over_green1 < red_over_green2:
         condition = "Route 1 is the shortest, flattest, and smoothest."
    elif route_dist1 < route_dist2 and el_range1 > el_range2 and red_over_green1 < red_over_green2:
         condition = "Route 1 is the smoothest and shortest, and Route 2 is the flattest."
    elif route_dist1 > route_dist2 and el_range1 > el_range2 and red_over_green1 < red_over_green2:
         condition = "Route 1 is the smoothest, and Route 2 is the flattest and shortest."
    elif route_dist1 < route_dist2 and el_range1 > el_range2 and red_over_green1 > red_over_green2:
         condition = "Route 1 is the shortest, and Route 2 is the flattest and smoothest."
    else:
         condition = "Route 1 is the shortest and flattest, and Route 2 is the smoothest."


    return render_template("output.html",map_code1 = map_code1, map_code2 = map_code2, center_point = center_point1,start_point = start_point,end_point = end_point,condition = condition)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
  # the_result = ModelIt(patient,births)
  # return render_template("output.html", births = births, the_result = the_result)
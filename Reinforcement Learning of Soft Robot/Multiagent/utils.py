'''
Script containing utility functions
'''

from typing import Dict, Tuple
import numpy as np
import cv2
import glob
import shutil
from pathlib import Path
import yaml
import os

def load_yaml(yaml_filepath:os.PathLike) -> Dict:
    '''
    Loads a YAML from provided filepath location
    '''
    assert Path(yaml_filepath).suffix=='yaml', "Provided YAML filepath must be a YAML!!"

    with open(yaml_filepath, 'r') as infile:
        yaml_dict = yaml.safe_load(infile)
    return yaml_dict

def eval_yaml(yaml_dict:Dict) ->Dict:
    '''
    Takes the loadad yaml file and evaluates any math that was provided
    '''

    for key, item in yaml_dict.items():
        if isinstance(item, str):
            try:
                yaml_dict[key] = eval(item)
            except:
                print(f'Unable to evaluate provided string: {item}')
    return yaml_dict

def calc_JAMoEBA_Radius(skinRadius:float, 
                        skinRatio:int, 
                        botRadius:float, 
                        numBots:int) -> float:
    """
    Inputs:
        - skinRadius (float): The radius of skin particles on system
        - skinRatio (int): Ratio of number of skin particles per bot
        - botRadius (float): The radius of bot particles on system
        - numBots (int): Number of bots in the system

    Returns:
        - R (float): Radius of the system given parameters
    """
    startDistance = skinRadius # The start distance between bots
    arcLength = 2*botRadius+skinRatio*(2*skinRadius)+(skinRatio+1)*startDistance
    theta = 2*np.pi/numBots
    R = arcLength/theta #**
    return R

def createVideo(saveLoc:os.PathLike, 
                imgLoc:os.PathLike, 
                videoName:str, 
                imgShape:Tuple[int,int]):
    '''
    Takes the images in provided location, imgLoc, and generates a video by 
    stitching together frames.
    '''
    out = cv2.VideoWriter(saveLoc+videoName+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 40, imgShape)
    for file in glob.glob(imgLoc+'*.jpg'):
        img = cv2.imread(file)
        out.write(img)
    out.release
    
    shutil.rmtree(imgLoc)
    print('Video Creation Complete')

def flatten(l):
    """
    Given a list that may contain arrays and scalars, will return an unwrapped list
    """
    for item in l:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
import numpy as np
import os
import sys
import pandas as pd
import json
from scipy import stats

import matplotlib.pylab as plt

sys.path.append(os.getcwd())
import functions.process as proc
import functions.waveletTransform as wt
import functions.algebra as al
from functions.io import makedir

## Settings and load data

def setup(inpath, outpath):

    ins = {}
    outs = {}
    for dirIn in inpath:
        in_name = os.path.basename(dirIn)
        out = makedir(os.path.join(outpath, f"{in_name}_engine"))

        for fn in os.listdir(dirIn):
            if "results" in fn:
                ins[fn]= os.path.join(dirIn, fn)
                outs[fn] = out
    return ins, outs



# Aim is to analyse the eigenpharynx of each results file
def FeatureEngine(data, outs, logger):
    for fn in data:
        logger.info(f"\nfeature calculation for {fn}")
        name = fn.split(".")[0]

        ### CLine_Concat is necessary as long as result files are in csv file format
        if 'csv' in fn:  
            PG = pd.read_csv(data[fn])
            if not 'Centerline' in PG.columns:
                logger.debug('!!! NO "Centerline" FOUND IN AXIS, MOVING ON TO NEXT VIDEO')
                continue
            CLine = proc.prepCL_concat(PG, "Centerline")
        elif 'json' in fn:
            PG = pd.read_json(data[fn])['Centerline']
            CLine = np.array([row for row in PG])
        CLine = CLine[:,:,::-1] ### VERY IMPORTANT, flips x and y of CenterLine, so that x is first

        #sanity check: inverted?
        CLine = proc.flip_ifinverted(CLine)

        ### feature calculation #####################################################################################
        XY = PG[['x','y']].values                                                                                 ### center of mass (x and y coordinates in PG)
        length, vec, angles = proc.angle(CLine[:,::np.ceil(CLine.shape[1]/34).astype(int)])                       ### calculates the vectors, their length and their inbetween angles of all centerline coordinates
        len_sum = np.sum(length, axis=1)                                                                          ### calculates the overall summed length of the pharynx
        ang_sum = np.sum(angles, axis =1)[:,np.newaxis]                                                           ### calculates the overall angle of the pharynx  

        adjustCL = (CLine-np.mean(CLine))+np.repeat(XY.reshape(XY.shape[0],1,XY.shape[1]), CLine.shape[1], axis=1) ### CenterLine in arena space reference for  following calculations # fits better than subtracting 50
        difflen, angle, v2_len = al.AngleLen(adjustCL, XY, hypotenuse = "v1", over="space", diffindex=[5,0])       ### vectors and angle between nosetip, defined as first 5 CLine points, and center of mass

        nose_unit_space, nose_len_space, nose_diff_space = al.unitvector_space(adjustCL,diffindex=[5,0])           ### unit vectors for calculation of angdiff and angspeed
        nose_unit_frames, nose_len_frame, nose_diff_frame = al.unitvector(adjustCL[:,0])                           ### unit vectors for calculation of angdiff and angspeed
        XY_unit, XY_len, XY_diff = al.unitvector(XY)                                                               ### unit vectors for calculation of angdiff and angspeed

        angdiff_nose_frames = np.arccos(nose_unit_frames[1:,0]*nose_unit_frames[:-1,0] + 
                                        nose_unit_frames[1:,1]*nose_unit_frames[:-1,1])                            ### angular difference between sequential (temporal) angles
        angspeed_nose_sec = al.angular_vel_dt(angdiff_nose_frames, dt=1)                                           ### circular mean velocity over dt: 1 sec, velocity: angular difference between frames

        crop = min(len(nose_unit_frames), len(nose_unit_space))
        dot_noses = nose_unit_frames[:crop,0]*nose_unit_space[:crop,0] +nose_unit_frames[:crop,1]*nose_unit_space[:crop,1]
        angle_noses = np.arccos(dot_noses) # mod of Vector is 1, so /mod can be left away
        angle, angdiff_nose_frames, angspeed_nose_sec, angle_noses = al.extendtooriginal((angle, 
                                                                                          angdiff_nose_frames, 
                                                                                          angspeed_nose_sec, 
                                                                                          angle_noses), 
                                                                                         (adjustCL.shape[0],1))

        ### concatanation ###########################################################################################
        new_data = pd.DataFrame(np.hstack((ang_sum, 
                                           len_sum,
                                           angle, 
                                           angdiff_nose_frames, 
                                           angspeed_nose_sec,
                                           angle_noses)), 
                                columns=['SumAngle', 
                                         'Length',
                                         'NosetipCmAngle',
                                         'AngDiffNoseFrames',
                                         'AngSpeedNose', 
                                         'AngBetweenNose'
                                        ])
        col_org_data = ['area', 
                        'velocity', 
                        'negskew_clean',
                        'cms_speed', 
                        'rate',
                       ]
        PG_new = pd.concat([PG[col_org_data], new_data], axis=1)
        col_ang = ['SumAngle', 'NosetipCmAngle','AngDiffNoseFrames','AngSpeedNose', 'AngBetweenNose']
        col_raw_data = PG_new.columns
        ### append smoothed #########################################################################################
        PG_new = pd.concat([PG_new,
                            PG_new[PG_new.columns.difference(PG_new[col_ang].columns)].rolling(window=60, min_periods=1, center=True).mean().add_suffix(f"_mean"),
                            PG_new[col_ang].rolling(window=60, min_periods=1, center=True).apply(stats.circmean, kwargs={'high':np.pi, "low":-np.pi}).add_suffix(f"_mean")], 
                            axis=1)

        ### Wavelet Transform #######################################################################################
        scales = np.linspace(3, 50, 10)
        for col in col_raw_data:
            lowpass_d = wt.lowpassfilter(PG_new[col].fillna(0).values/PG_new[col].mean(), 0.01)
            coefficients, frequencies = wt.cwt_signal(lowpass_d, scales)#rec[:-1]
            maxfreq_idx = np.argmax(abs(coefficients), axis=0)
            maxfreq = maxfreq_idx.copy().astype('float')
            for i, f in enumerate(frequencies):
                np.put(maxfreq, np.where(maxfreq_idx == i), [f])

            #lowpass_df = pd.DataFrame(lowpass_d, columns=[f'{col}_lowpass'])
            cols_coeff = pd.DataFrame(np.stack((abs(coefficients)), axis=1), columns=[f'{col}_cwt%02d'% scl for scl in scales])
            maxfreq_df = pd.DataFrame(maxfreq, columns=[f'{col}_maxfreq'])

            PG_new = pd.concat([PG_new, 
                                #lowpass_df,
                                cols_coeff, maxfreq_df], axis=1)

        ### Wavelet Transform #######################################################################################
        ### Save to specified output  ###############################################################################
        jsnL = json.loads(PG_new.to_json(orient="split"))
        jsnF = json.dumps(jsnL, indent = 4)
        outs[fn] = os.path.join(outs[fn],f"{name}_features.json")
        with open(outs[fn], "w") as outfile:
            outfile.write(jsnF)
    return outs

def run(inpath, outpath, logger):
    if isinstance(inpath,str):
        inpath = [inpath]
    ins, outs = setup(inpath, outpath)
    outs = FeatureEngine(ins, outs, logger)
    return outs
import numpy as np
import os
import sys
import pandas as pd
import json
from scipy import stats
import tqdm

import matplotlib.pylab as plt

sys.path.append(os.getcwd())
import functions.process as proc
import functions.waveletTransform as wt
import functions.algebra as al
from functions.io import makedir

## Settings and load data

def setup(inpath, outpath, skip_already):
    engine_done = []
    ins = {}
    outs = {}
    for dirIn, dirOut in zip(inpath,outpath):
        in_name = os.path.basename(dirIn)
        out = makedir(dirOut)
        if skip_already:
            engine_done = [f for f in os.listdir(out)]
        for fn in os.listdir(dirIn):
            out_fn = os.path.join(out, f"{fn.split('.')[0]}_features.json")
            if "labeldata" in fn and not fn[0] == '.' and not out_fn in engine_done:
                ins[fn] = os.path.join(dirIn, fn)
                outs[fn] = out_fn
    return ins, outs



# Aim is to analyse the eigenpharynx of each results file
def FeatureEngine(data, outs, logger, skip_engine, fps=30):
    XYs, CLines = {},{}
    for fn in tqdm.tqdm(data):
        logger.info(f"\nfeature calculation for {fn}")
        name = fn.split(".")[0]

        ### CLine_Concat is necessary as long as result files are in csv file format
        if 'csv' in fn:  
            PG = pd.read_csv(data[fn])
            if not 'Centerline' in PG.columns:# or 'reversals_nose' not in PG.columns:
                logger.debug('!!! NO "Centerline" FOUND IN AXIS, MOVING ON TO NEXT VIDEO')
                continue
            CLine = proc.prepCL_concat(PG, "Centerline")
        elif 'json' in fn:
            PG = pd.read_json(data[fn])['Centerline']
            CLine = np.array([row for row in PG])
        CLine = CLine[:,:,::-1] ### VERY IMPORTANT, flips x and y of CenterLine, so that x is first

        # look for large area, filter
        large_area = np.where(PG['area']>=np.mean(PG.area)*1.5)[0]
        large_diff = [0]+[i+1 for i,e in enumerate(np.diff(large_area)) if e > 1] if large_area.size > 0 else []
        large_size = np.diff(large_diff, append=len(large_area))
        large_ranges = [range(large_area[d_i], large_area[d_i]+s) for d_i, s in zip(large_diff, large_size)]
        logger.info(f'Area larger than threshold, collision assumed in {large_ranges}.\nCalculation of features will be done in splits, ignoring those and adjacent* ranges. *That are less than 1 sec long.')

        correct_area = np.where(PG['area']<=np.mean(PG.area)*1.5)[0]
        correct_diff = [0]+[i+1 for i,e in enumerate(np.diff(correct_area)) if e > 1]
        correct_size = np.diff(correct_diff, append=len(correct_area))
        data_splits = []
        for i in range(len(correct_size)):
            if correct_size[i] > fps:
                correct_start = correct_area[correct_diff[i]]
                correct_end = correct_area[correct_diff[i]] + correct_size[i]
                data_splits.append([correct_start, correct_end])

        PG_splits = []
        CLine_splits = np.empty((0,*CLine.shape[1:]))
        XY_splits = np.empty((0,2))

        ### calculate features for each split independently
        for iter,(on, off) in enumerate(data_splits):
            logger.info(f'split {iter}, range: {on, off}')
            
            ### open data for current split
            PG_split = PG.iloc[on:off].reset_index(drop=True)
            XY_split = PG_split[['x','y']].values
            ### tries to identify frames where the Centerline is tracked up side down, checks coherence of movement
            CLine_split = proc.flip_ifinverted(CLine[on:off], XY_split)
            ### created CL in arena space; mean fits better than subtracting 50, which would be half of set length
            adjustCL_split = (CLine_split-np.mean(CLine_split))+np.repeat(XY_split.reshape(XY_split.shape[0],1, XY_split.shape[1]), 
                                                                          CLine_split.shape[1], axis=1)
            
            ### feature calculation #####################################################################################
            if not skip_engine:
                ### calculates the vectors, their length and their inbetween angles of all centerline coordinates
                length, _, CLArctan = proc.angle(CLine_split[:,::np.ceil(CLine_split.shape[1]/34).astype(int)])
                ### calculates the overall summed length of the pharynx
                SumLen = np.sum(length, axis=1)
                ### calculates the curvature of the pharynx
                Curvature = al.TotalAbsoluteCurvature(CLArctan, length)
    
    
                ### vectors and angle between nosetip, defined as first 5 CLine_split points, and center of mass
                _, tip2cm_arccos, _ = al.AngleLen(adjustCL_split, XY_split, hypotenuse = "v1", over="space", diffindex=[5,0])
                _, tip2tip_arccos, _ =  al.AngleLen(adjustCL_split[:,0], hypotenuse = "v1", over="frames")
                _, tip2tipMv_arccos, _ = al.AngleLen(adjustCL_split, adjustCL_split[:,0], hypotenuse = "v1", over="space", diffindex=[5,0])
                #angspeed_nose_sec = al.angular_vel_dt(tip2tip_arccos, dt=1)
                
                ### calculate reversal, as over 120deg 
                reversal_bin = np.where(tip2cm_arccos >= np.deg2rad(120), 1, 0)
                reversal_events = np.clip(np.diff(reversal_bin, axis=0), 0, 1)
                reversal_fract = pd.Series(reversal_bin.squeeze()).rolling(30, center=True).apply(lambda w: np.mean(w))
                reversal_rate = pd.Series(reversal_events.squeeze()).rolling(30, center=True).apply(lambda w: np.mean(w)*30)
    
    
                ### reshapes all features to fit the original
                Curvature, tip2cm_arccos, tip2tip_arccos, tip2tipMv_arccos, reversal_rate, reversal_fract = al.extendtooriginal((Curvature, tip2cm_arccos, tip2tip_arccos, tip2tipMv_arccos, 
                                                                                                                                 reversal_rate, reversal_fract), (adjustCL_split.shape[0],1))
                ### hstack all calculated features 
                new_data = pd.DataFrame(np.hstack((Curvature, SumLen, tip2cm_arccos, tip2tip_arccos, tip2tipMv_arccos, reversal_rate, reversal_fract)), 
                                        columns=['Curvature', 'Length','tip2cm_arccos','tip2tip_arccos', 'tip2tipMv_arccos','reversal_rate', 'reversal_fract'])
    
    
                ### load original data from PharaGlow results file
                col_org_data = ['area', 'velocity', 'rate','negskew_clean',]####################################
                col_org_notexist = [c not in PG_split.columns for c in col_org_data]
                if any(col_org_notexist):
                    logger.debug(f'WARNING {list(itertools.compress(col_org_data,col_org_notexist))} not in data')
                    ### TODO make entries with nans
                    continue
    
    
                ### combine new and original features in one Df
                PG_new = pd.concat([PG_split[col_org_data], new_data], axis=1)
                col_raw_data = PG_new.columns
    
                ### Calculating smooth, freq and amplitude for all columns
                scales = np.linspace(3, 50, 10)
                #freq = np.logspace(np.log10(0.3), np.log10(4.5), 10)####################################
                #scales = np.floor(15/freq)####################################
                for col in  PG_new.columns:
                    lowpass_d = wt.lowpassfilter(PG_new[col].fillna(0).values/PG_new[col].mean(), 0.01)
                    lowpass_toolarge = PG_new.shape[0]-lowpass_d.shape[0]
                    if lowpass_toolarge < 0:
                        lowpass_d = lowpass_d[:lowpass_toolarge] 
    
                    coefficients, frequencies = wt.cwt_signal(lowpass_d, scales)#rec[:-1]
                    maxfreq_idx = np.argmax(abs(coefficients), axis=0)
                    maxfreq = maxfreq_idx.copy().astype('float')
                    for i, f in enumerate(frequencies):
                        np.put(maxfreq, np.where(maxfreq_idx == i), [f])
    
                    cols_coeff = pd.DataFrame(np.stack((abs(coefficients)), axis=1), columns=[f'{col}_cwt%02d'% scl for scl in scales])
                    maxfreq_df = pd.DataFrame(maxfreq, columns=[f'{col}_maxfreq'])
    
                    PG_new = pd.concat([PG_new, cols_coeff, maxfreq_df], axis=1)

        
                ### encode angular columns as cos sin
                deg_cols = PG_new.filter(regex='arctan$|arccos$').columns
                cos_ = al.encode_cos(PG_new[deg_cols]).rename(columns = lambda s: s.replace(s, s.split('_')[0]+'_cos'))
                sin_ = al.encode_sin(PG_new[deg_cols]).rename(columns = lambda s: s.replace(s, s.split('_')[0]+'_sin'))
                ### concat encoded columns, drop angular columns # for ease of distance and mean calculation
                PG_new = pd.concat([PG_new, cos_, sin_], axis=1).drop(deg_cols, axis=1)
    
                ### calculate means of the basal columns (not cwt or maxfreq)
                col_basic = PG_new.columns.drop(list(PG_new.filter(regex='cwt|maxfreq').columns))
                PG_new = pd.concat([PG_new, PG_new[col_basic].rolling(window=60, min_periods=1, center=True).mean().add_suffix(f"_mean")], axis=1)
    
                ### set index to original split
                PG_new = PG_new.set_index(pd.Index(range(on,off)), drop=True)
    
                ### vstack with padding of nans in range of next large area, not needed for dfs as they contain index
                PG_splits.append(PG_new)
            prepadCL = np.full((on-len(CLine_splits), *CLine_split.shape[1:]), np.nan)
            prepadXY = np.full((on-len(XY_splits), *XY_split.shape[1:]), np.nan)
            CLine_splits = np.vstack([CLine_splits,prepadCL,CLine_split])
            XY_splits = np.vstack([XY_splits,prepadXY,XY_split])

        if not skip_engine:
            ### concat df splits, reindex to include nan in areas not present in PG_splits indices
            PG_joined = pd.concat(PG_splits)
            correct_idx = PG_joined.index
            PG_joined = PG_joined.reindex(pd.Index(range(0,len(PG))))
    
            # interpolate features, also newly calculated at previously detected large area glitches
            PG_joined.interpolate('ffill', axis=0)
            logger.info('Ffill-Interpolation of nan frames')
            # if glitch longer than 1 sec reset nans
            glitch_idx = PG_joined.index.difference(correct_idx)
            glitch_diff = [0]+[i+1 for i,e in enumerate(np.diff(glitch_idx)) if e > 1]
            glitch_size = np.diff(glitch_diff, append=len(glitch_idx))
            for i in range(len(glitch_size)):
                if glitch_size[i] > fps:
                    glitch_start = glitch_idx[glitch_diff[i]]
                    glitch_end = glitch_idx[glitch_diff[i]] + glitch_size[i]
                    logger.info(f'Exempted from interpolation: {range(glitch_start,glitch_end)} (over 1 sec long)')
                    PG_joined.iloc[glitch_start:glitch_end] = np.nan
    
    
            jsnL = json.loads(PG_joined.to_json(orient="split"))
            jsnF = json.dumps(jsnL, indent = 4)
            outs[fn] = os.path.join(outs[fn])
            with open(outs[fn], "w") as outfile:
                outfile.write(jsnF)
                
        ### postpad the end
        CL_joined = np.vstack([CLine_splits, np.full((len(CLine) - off, *CLine_splits.shape[1:]), np.nan)])
        XY_joined = np.vstack([XY_splits, np.full((len(PG) - off, *XY_splits.shape[1:]), np.nan)])
        XYs[fn] = XY_joined
        CLines[fn] = CL_joined
    return outs, XYs, CLines
    
def run(inpath, outpath, logger, return_XYCLine = False, skip_already = False, skip_engine = False, fps=30):
    if isinstance(inpath,str):
        inpath = [inpath]
    if isinstance(outpath,str):
        outpath = [outpath]
    ins, outs = setup(inpath, outpath, skip_already)
    outs, XYs, CLines = FeatureEngine(ins, outs, logger, skip_engine, fps)
    if return_XYCLine:
        return XYs, CLines
import numpy as np
import os
import sys
import pandas as pd
import json
from scipy import stats
import tqdm
import itertools

import matplotlib.pylab as plt

sys.path.append(os.getcwd())
import functions.process as proc
import functions.waveletTransform as wt
import functions.algebra as al
from functions.io import makedir

sys.path.append(os.path.expanduser("~"))
from PpaPy.calculate.descriptive import velocity

## Settings and load data
class Printlogger():
    def info(str):
        print(str)

class FeatureScaler():
    def __init__(self, path_container, features, quantile=.95, json_orient='split', logger=Printlogger):
        self.path = path_container
        self.features = features
        self.quantile = quantile
        self.json_orient = json_orient
        self.logger = logger
        self.list2dict()
    
    def list2dict(self):
        if isinstance(self.path, list):
            self.path = {os.path.basename(p): p for p in self.path}
    
    def feature_quantile(self):
        data = pd.DataFrame([])
        features = self.features.keys()
        for fn in self.path:
            self.logger.info(f'feature_quantile: {fn}')
            if fn.endswith('csv'):
                PG = pd.read_csv(self.path[fn])
            elif fn.endswith('json'):
                PG = pd.read_json(self.path[fn], orient=self.json_orient)
            data = pd.concat([data, PG[features]])
        
        data_quantiles = data.quantile(self.quantile).to_dict()
        return data_quantiles
    
    def feature_scaler(self):
        """
        Input:
            wanted_quantile (dict): dict containing feature names (keys) and quantile that should be scaled to, e.g. from WT data
        Returns:
            scaling values for data (dict)
        """
        data_quantile = self.feature_quantile()
        scalings = {k: self.features[k]/v for k,v in data_quantile.items()}
        
        return data_quantile, scalings
    
    def scale(self, scalings=None):
        # scale features by scaling value provided through scalings
        if scalings is None:
            data_quantile, scalings = self.feature_scaler()
        self.logger.info(f"\{int(self.quantile*100)} quantiles before scaling:\n{data_quantile}\n")
        self.logger.info(f"\nScaling Factor:\n{scalings}\n")

        for fn in self.path:
            fdata = pd.read_json(self.path[fn], orient=self.json_orient)
            for feat in scalings:
                fdata[feat] = fdata[feat]*scalings[feat]

            # write
            jsnL = json.loads(fdata.to_json(orient=self.json_orient))
            jsnF = json.dumps(jsnL, indent = 4)
            with open(self.path[fn], "w") as outfile:
                self.logger.info("writing scaled version to file.")
                outfile.write(jsnF)
            
        scaled_quantile = self.feature_quantile()
        self.logger.info(f"\{int(self.quantile*100)} quantiles after scaling:\n{scaled_quantile}\n")

class CalculateFeatures():
    
    def __init__(self, inpath, outpath, logger=None, skip_already = False, fps=30, break_after=False, out_fn_suffix='features', inpath_with_subfolders=False):
        self.inpath = inpath
        self.outpath = outpath
        self.logger = logger
        self.skip_already = skip_already 
        self.fps = fps
        self.break_after = break_after
        self.out_fn_suffix = out_fn_suffix
        self.inpath_with_subfolders = inpath_with_subfolders

        if isinstance(self.inpath,str):
            self.inpath = [self.inpath]
        if isinstance(self.outpath,str):
            self.outpath = [self.outpath]
        
        if self.logger is None:
            self.logger = Printlogger

        self.setup()
        #outs, XYs, CLines = self.FeatureEngine(ins, outs)
        

    def setup(self):
        engine_done = []
        self.ins = {}
        self.outs = {}
        for dirIn, dirOut in zip(self.inpath,self.outpath):
            out = makedir(dirOut)
            if self.skip_already:
                engine_done = ['_'.join(f.split('_')[:-1]) for f in os.listdir(out)]
            i = 0
            for fn in os.listdir(dirIn):
                if self.inpath_with_subfolders:
                    subf = os.path.basename(dirIn)
                    uniq_fn = '_'.join([subf, fn])
                else:
                    uniq_fn = fn

                id = uniq_fn.split('.')[0]
                out_fn = f"{id}_{self.out_fn_suffix}.json"
                print(id)
                if "labeldata" in fn and not fn[0] == '.' and not id in engine_done:
                    self.ins[uniq_fn] = os.path.join(dirIn, fn)
                    self.outs[uniq_fn] = os.path.join(out, out_fn)
                    if self.break_after:
                        if i == self.break_after:
                            break
                        i += 1

            
    
    def load_data(self, fn, inpath_dict=None):
        if inpath_dict is None:
            inpath_dict = self.ins
            
        ### CLine_Concat is necessary as long as result files are in csv file format
        if fn.endswith('csv'):
            PG = pd.read_csv(inpath_dict[fn])
            if not 'Centerline' in PG.columns:# or 'reversals_nose' not in PG.columns:
                self.logger.debug(f'!!! NO "Centerline" FOUND IN AXIS, MOVING ON TO NEXT VIDEO\n')
                return None, None
            CLine = proc.prepCL_concat(PG, "Centerline")
        if fn.endswith('json'):
            PG = pd.read_json(inpath_dict[fn])['Centerline']
            CLine = np.array([row for row in PG])
        CLine = CLine[:,:,::-1] ### VERY IMPORTANT, flips x and y of CenterLine, so that x is first
        return PG, CLine

    # Aim is to analyse the eigenpharynx of each results file
    def run(self, return_XYCLine = False, skip_engine = False, ):
        col_org_data = ['area', 'rate','negskew_clean',]
        scales = np.linspace(3, 50, 10)
        negskew_scale = 15/np.linspace(.3,5,10)
        
        XYs, CLines = {},{}
        for fn in tqdm.tqdm(self.ins):
            self.logger.info(f"feature calculation for {fn}")

            PG, CLine = self.load_data(fn)
            if PG is None:
                continue

            # look for large area, filter
            large_area = np.where(PG['area']>=np.mean(PG.area)*1.5)[0]
            large_diff = [0]+[i+1 for i,e in enumerate(np.diff(large_area)) if e > 1] if large_area.size > 0 else []
            large_size = np.diff(large_diff, append=len(large_area))
            large_ranges = [range(large_area[d_i], large_area[d_i]+s) for d_i, s in zip(large_diff, large_size)]
            self.logger.info(f'Area larger than threshold, collision assumed in {large_ranges}.\nCalculation of features will be done in splits, ignoring those and adjacent* ranges. *That are less than 1 sec long.')

            correct_area = np.where(PG['area']<=np.mean(PG.area)*1.5)[0]
            correct_diff = [0]+[i+1 for i,e in enumerate(np.diff(correct_area)) if e > 1]
            correct_size = np.diff(correct_diff, append=len(correct_area))
            data_splits = []
            for i in range(len(correct_size)):
                if correct_size[i] > self.fps:
                    correct_start = correct_area[correct_diff[i]]
                    correct_end = correct_area[correct_diff[i]] + correct_size[i]
                    data_splits.append([correct_start, correct_end])

            PG_splits = []
            CLine_splits = np.empty((0,*CLine.shape[1:]))
            XY_splits = np.empty((0,2))

            ### calculate features for each split independently
            for iter,(on, off) in enumerate(data_splits):
                self.logger.info(f'split {iter}, range: {on, off}')
                
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
                    ### vectors and angle between nosetip, defined as first 5 CLine_split points, and center of mass
                    _, tip2cm_arccos, _ = al.AngleLen(adjustCL_split, XY_split, hypotenuse = "v1", over="space", v1_args=dict(diffindex=[5,0]))

                    # anterior vs posterior part
                    _, tip2back_arccos, _ = al.AngleLen(adjustCL_split, adjustCL_split, hypotenuse = "v1", over="space", v2_over='space', v1_args=dict(diffindex=[40,10]),v2_args=dict(diffindex=[-10,60]))
                    # cm vs cm over time
                    _, cl2cl_arccos_dt30, _ = al.AngleLen(adjustCL_split, hypotenuse = "v1", over="space", v1_args=dict(diffindex=[-10,10]), v2_diff=30)

                    # pharynx vs cm over time
                    _, movedir, _ = al.AngleLen(adjustCL_split, XY_split, hypotenuse = "v1", over='space', v1_args=dict(diffindex=[-10,10]))
                    reversal_bin = np.where(pd.DataFrame(movedir).rolling(10).mean() >= np.deg2rad(120), 1, 0)
                    reversal_events = np.clip(np.diff(reversal_bin, axis=0), 0, 1)
                    reversal_rate = pd.Series(reversal_events.squeeze()).rolling(30, center=True).apply(lambda w: np.mean(w)*30)

                    # assume that v2 (cm vector) is hypotenuse, makes sideways headmotion vector more realistic
                    # pharynx vs cm over time, with sin calculated
                    headmotion, leftright, _ = al.AngleLen(adjustCL_split, XY_split, hypotenuse = "v2", over='space', v2_args=dict(diff_step=1), angletype=np.arcsin)
                    leftright_abs_smooth = abs(pd.DataFrame(leftright).rolling(30).mean())
                    headvelocity = abs(headmotion[:-1] - headmotion[1:])/(1/30)
        
                    # edited
                    ### reshapes all features to fit the original
                    tip2cm_arccos, tip2back_arccos, cl2cl_arccos_dt30, movedir, reversal_bin, reversal_rate, headmotion, headvelocity, leftright_abs_smooth = al.extendtooriginal((
                                                                        tip2cm_arccos,
                                                                        tip2back_arccos, 
                                                                        cl2cl_arccos_dt30, 
                                                                        movedir, 
                                                                        reversal_bin, 
                                                                        reversal_rate, 
                                                                        headmotion, 
                                                                        headvelocity, 
                                                                        leftright_abs_smooth
                                                                        ), 
                                                                        (adjustCL_split.shape[0],1))

                    # edited
                    # hstack all calculated features 
                    new_data = pd.DataFrame(np.hstack((tip2cm_arccos, tip2back_arccos, cl2cl_arccos_dt30, movedir, reversal_bin, reversal_rate, leftright_abs_smooth, headmotion, headvelocity)), 
                                            columns=['tip2cm_arccos', 'tip2back_arccos', 'cl2cl_arccos_dt30', 'movedir', 'reversal_bin', 'reversal_rate', 'leftright_abs_smooth', 'headmotion', 'headvelocity'])
            
                    ### load original data from PharaGlow results file
                    col_org_notexist = [c not in PG_split.columns for c in col_org_data]
                    if any(col_org_notexist):
                        self.logger.debug(f'WARNING {list(itertools.compress(col_org_data,col_org_notexist))} not in data\n')
                        ### TODO make entries with nans
                        continue
        
        
                    ### combine new and original features in one Df
                    PG_new = pd.concat([PG_split[col_org_data], new_data], axis=1)
                    PG_new['velocity'] = velocity(PG_split['x_scaled'],PG_split['y_scaled'], 1, fps=self.fps, dt=30)
                    PG_new['velocity_mean'] = PG_new['velocity'].rolling(window=60, min_periods=1).mean()
                    PG_new['velocity_dt60'] = velocity(PG_split['x_scaled'],PG_split['y_scaled'], 1, fps=self.fps, dt=60)
                    PG_new['velocity_dt150'] = velocity(PG_split['x_scaled'],PG_split['y_scaled'], 1, fps=self.fps, dt=150)
        
                    ### Calculating smooth, freq and amplitude for all columns
                    for col in  PG_new.columns:
                        if 'negskew'in col:
                            _scale = negskew_scale
                        else:
                            _scale = scales
                        lowpass_d = wt.lowpassfilter(PG_new[col].fillna(0).values/PG_new[col].mean(), 0.01)
                        lowpass_toolarge = PG_new.shape[0]-lowpass_d.shape[0]
                        if lowpass_toolarge < 0:
                            lowpass_d = lowpass_d[:lowpass_toolarge] 
        
                        coefficients, frequencies = wt.cwt_signal(lowpass_d, _scale)#rec[:-1]
                        maxfreq_idx = np.argmax(abs(coefficients), axis=0)
                        maxfreq = maxfreq_idx.copy().astype('float')
                        for i, f in enumerate(frequencies):
                            np.put(maxfreq, np.where(maxfreq_idx == i), [f])
        
                        # edited
                        maxfreq_df = pd.DataFrame(maxfreq, columns=[f'{col}_maxfreq'])
                        if 'velocity_dt60' in col or 'negskew' in col:
                            cols_coeff = pd.DataFrame(np.stack((abs(coefficients)), axis=1), columns=[f'{col}_cwt%03.2f'% fr for fr in np.round(frequencies, 2)]) # type: ignore
                            PG_new = pd.concat([PG_new, cols_coeff, maxfreq_df], axis=1)
                        else:
                            PG_new = pd.concat([PG_new, maxfreq_df], axis=1)

            
                    # TODO: drop angular columns or make sure during feature selection those are dropped!
                    # drop arccos   
                    #deg_cols = PG_new.filter(regex='arccos$').columns
                    #PG_new = PG_new.drop(deg_cols, axis=1)

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
                self.logger.info('Ffill-Interpolation of nan frames')
                # if glitch longer than 1 sec reset nans
                glitch_idx = PG_joined.index.difference(correct_idx)
                glitch_diff = [0]+[i+1 for i,e in enumerate(np.diff(glitch_idx)) if e > 1]
                glitch_size = np.diff(glitch_diff, append=len(glitch_idx))
                for i in range(len(glitch_size)):
                    if glitch_size[i] > self.fps:
                        glitch_start = glitch_idx[glitch_diff[i]]
                        glitch_end = glitch_idx[glitch_diff[i]] + glitch_size[i]
                        self.logger.info(f'Exempted from interpolation: {range(glitch_start,glitch_end)} (over 1 sec long)')
                        PG_joined.iloc[glitch_start:glitch_end] = np.nan

                jsnL = json.loads(PG_joined.to_json(orient="split"))
                jsnF = json.dumps(jsnL, indent = 4)
                #self.outs[fn] = os.path.join(self.outs[fn])
                with open(self.outs[fn], "w") as outfile:
                    outfile.write(jsnF)
                    
            ### postpad the end
            CL_joined = np.vstack([CLine_splits, np.full((len(CLine) - off, *CLine_splits.shape[1:]), np.nan)])
            XY_joined = np.vstack([XY_splits, np.full((len(PG) - off, *XY_splits.shape[1:]), np.nan)])
            XYs[fn] = XY_joined
            CLines[fn] = CL_joined
        
            self.logger.info(f"\n")

        if return_XYCLine:
            return XYs, CLines
        return self.outs, XYs, CLines
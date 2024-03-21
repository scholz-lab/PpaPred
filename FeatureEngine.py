import numpy as np
import os
import sys
import pandas as pd
import json
from scipy import stats
import pywt
import logging

import matplotlib.pylab as plt

def extendtooriginal(arrays, org_shape):
    extended = []
    for arr in arrays:
        nans = np.full(org_shape, np.nan)
        nans.flat[:arr.size] = arr
        extended.append(nans)
    return extended

def unitvector(xyarray):
    base = np.diff(xyarray, axis=0)
    vlen = np.linalg.norm(base, axis = 1).reshape(base.shape[0],1)
    unit_vec = np.divide(base,vlen)
    unit_vec = np.nan_to_num(unit_vec)
    return unit_vec, vlen, base

def unitvector_space(xyarray, diffindex=[0,99]):
    base = xyarray[:,diffindex[1]]-xyarray[:,diffindex[0]]
    vlen = np.linalg.norm(base, axis = 1).reshape(base.shape[0],1)
    unit_vec = np.divide(base,vlen)
    unit_vec = np.nan_to_num(unit_vec)
    return unit_vec, vlen, base

def AngleLen (v1, v2=None, hypotenuse = "v1", over="frames",**args):
    if over == "frames":
        unitfunction = unitvector
    elif over == "space":
        unitfunction = unitvector_space
        
    v1_unit, v1_len, v1_diff = unitfunction(v1,**args)
    if not v2 is None:
        v2_unit, v2_len, v1_diff = unitvector(v2)
    else:
        v2_unit, v2_len, v1_diff = v1_unit[1:], v1_len[1:], v1_diff[1:]
    
    hyp = {"v1":v1_len, "v2":v2_len}
    hyplen = hyp[hypotenuse]
    
    crop = min(len(v1_unit), len(v2_unit))
    #x1, y1, x2, y2 = v1_unit[:crop,0], v1_unit[:crop,1], v2_unit[:crop,0], v2_unit[:crop,1]
    dotProduct = v1_unit[:crop,0]*v2_unit[:crop,0] +v1_unit[:crop,1]*v2_unit[:crop,1]
    arccos = np.arccos(dotProduct) # mod of Vector is 1, so /mod can be left away  #arccos
    #arcsin = np.arcsin(dotProduct)
    
    difflen = np.multiply(np.sin(arccos[:crop]).flatten(),hyplen[:crop].flatten())
    
    return difflen, arccos, v1_diff

def TotalAbsoluteCurvature(Phi_i, L_i, axis=1):
    """
    Calcultes total absolute curvature of discrete curves, given the vector angle by Phi_i and the vector length by L_i
    """
    A_i= ((L_i[:,:-1]+L_i[:,1:])/2)
    k_i = Phi_i.squeeze()/A_i.squeeze()
    k = np.sum(abs(k_i), axis=axis)
    return k
    
def angle2vec(nose_unit_frames, nose_unit_space):
    crop = min(len(nose_unit_frames), len(nose_unit_space))
    dot_noses = nose_unit_frames[:crop,0]*nose_unit_space[:crop,0] +nose_unit_frames[:crop,1]*nose_unit_space[:crop,1]
    angle_noses = np.arccos(dot_noses) # mod of Vector is 1, so /mod can be left away
    return angle_noses
    
# encode cos, sin
def encode_cos(x):
    return np.cos(x)
def encode_sin(x):
    return np.sin(x)
    
###########################################################
def prepCL_concat (data, col):
    CenterLine = data[col]
    CLine_allframes = np.empty((len(CenterLine),100,2))
    for frame in range(len(CenterLine)):
        # accessing Centerline
        CLine = CenterLine[frame].replace('[[','').replace(']]','')
        CLine = CLine.split("], [")
        CLine_lst = []
        for i in CLine:
            lst = i.split(", ")
            lst = [float(_str) for _str in lst]
            CLine_lst.append(lst)
        CLine_arr = np.array(CLine_lst) #Correct assumption? col 0 is x?

        '''
        if frame % 4000 == 0:
            print(frame)
            print(CLine_arr[:5,])
            plt.scatter(CLine_arr[:,0],CLine_arr[:,1])
            plt.show()
        '''

        CLine_allframes[frame] = CLine_arr
        
    return CLine_allframes

def angle(Cl):
    """Use centerline points to calculate relative angles.
    Cl = (Nsamples, 100, 2) array
    """
    # vector components for relative vectors
    vec = np.diff(Cl, axis = 1)
    # normalize vectors. calculates the length of vectors
    length = np.linalg.norm(vec, axis = 2).reshape(vec.shape[0],vec.shape[1],1)
    unit_vec = vec/length
    # define vectors to calculate angles from: shift
    vec0 = unit_vec[:,:-1] # from 0 to forelast
    vec1 = unit_vec[:,1:] # from 1 to last
    # angles
    angles = np.zeros((len(length), vec0.shape[1])) # removed -1 in vec0.shape[1]-1
    for i in range(len(length)):
        last = 0
        for k in range(vec0.shape[1]): # removed -1 in vec0.shape[1]-1
            alpha = np.arctan2(vec0[i,k,1],vec0[i,k,0])-np.arctan2(vec1[i,k,1],vec1[i,k,0])
            
            if alpha > np.pi or alpha < -np.pi:
                while alpha < last-np.pi: alpha += 2*np.pi #check for out of range pi to -pi
                while alpha > last+np.pi: alpha -= 2*np.pi
            last = alpha
            
            angles[i,k] = alpha
    return length,vec,angles

def flip_ifinverted(arr, XY):
    # arr must have shape nframes,nCenterlinePoints, XYpoints
    # sanity check to see if orientation of centerline is coherent between frames
    # using shifted arrays to compare between cuurent frame and next
    # the initial comparison has to be adjusted however, because a inverted frame inverts the boolean value up to the next right-side-up frame
    # corrected boolean array is used to reorientate 

    # create shifted arrays to compare between frames
    tipnow = np.median(arr[:-1,:5],axis=1)
    tailnow = np.median(arr[:-1,-5:],axis=1)
    tipnext = np.median(arr[1:,:5],axis=1)
    tailnext = np.median(arr[1:,-5:],axis=1)
    
    # calculate euclidean distance between tip and next frame tip and tip and next frame tail
    # basis for decision which frame to flip: if tail is closer than tip, must be inverted
    tip2tip = np.linalg.norm(tipnow - tipnext, axis=1)
    tip2tail = np.linalg.norm(tipnow - tailnext, axis=1)
    tail2tail = np.linalg.norm(tailnow - tailnext, axis=1)
    tail2tip = np.linalg.norm(tailnow - tipnext, axis=1)
    same2 = tip2tip + tail2tail
    diff2 = tip2tail + tail2tip
    flipped = np.insert(same2 > diff2, 0, False)

    # flip True and False after each True until next True (incl.) (inverted comparison)
    # if inversion is present, flipped is based on current tail, up until next inversion
    wrongcompare = (np.array(np.where(flipped))[0]+1)
    if len(wrongcompare) % 2 != 0:
        wrongcompare = np.append(wrongcompare,len(flipped)+1)
    wrongcompare = wrongcompare.reshape(len(wrongcompare)//2,2)
    # boolean inversion of the wrong compared ones
    for i,j in wrongcompare:
        flipped[i:j] = ~flipped[i:j]

    if np.any(flipped):
        print('Following frames seem to be tracked upside down\nWe are going to flip those back:')
        print(np.where(flipped))

    # inversion
    arr[flipped] = arr[flipped,::-1]

    # test if movement direction alligns with Centerline orientation most of the time
    uCL,lCL,bCL = unitvector_space(arr, [-1,0])
    uXY, lXY, bXY = unitvector(XY)
    aCL = angle2vec(uCL,uXY)
    if np.mean(aCL) > np.pi/2:
        arr = arr[:,::-1]
        print('Worm seems to travel backwards the majority of time, flipped all centerlines to agree with forward travelling assumption')
    
    return arr

##################################################
def cwt_signal(signal, scales, dt=1/30, 
                 waveletname = 'gaus5', #cgau5
              ):
    """scales: defines the width of a wavelet. Related to frequency of waves. Since waves have diverse forms and are located in time, it is better to speak of pseudo frequencies
    scales are inversly related to frequencies. the fuction pywt.scale2frequency can be used to determine the frequency. But returns frequency for sampling period -> *30 if 30fps -> Hz
    depending on wavelet and dt, scales lead to different frequencies: for gaus5 and 30 fps scale of 1 -> 15 Hz, scale 10 -> 1.5 Hz
    scales: array like
    signal: array like
    
    returns coefficients in imaginary numbers, make absolute to compute further with
    """
    coefficients, frequencies = pywt.cwt(signal, scales, waveletname, dt)
    
    return coefficients, frequencies

def lowpassfilter(signal, thresh = 0.63, wavelet="db8"):
    """signal: array like, without nan values
    """
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

################################################### Settings and load data
def setup_logger(name, filename, level=logging.INFO, logformat ='%(asctime)s %(levelname)s %(message)s'):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    handler = logging.FileHandler(filename)        
    handler.setFormatter(logging.Formatter(logformat))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def makedir(dirname, basepath = ''):
    dirpath = os.path.join(basepath, dirname)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath
##################################################
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
def FeatureEngine(data, outs, logger, fps=30):
    XYs, CLines = {},{}
    for fn in data:
        logger.info(f"\nfeature calculation for {fn}")
        name = fn.split(".")[0]

        ### CLine_Concat is necessary as long as result files are in csv file format
        if 'csv' in fn:  
            PG = pd.read_csv(data[fn])
            if not 'Centerline' in PG.columns:# or 'reversals_nose' not in PG.columns:
                logger.debug('!!! NO "Centerline" FOUND IN AXIS, MOVING ON TO NEXT VIDEO')
                continue
            CLine = prepCL_concat(PG, "Centerline")
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
            CLine_split = flip_ifinverted(CLine[on:off], XY_split)
            ### created CL in arena space; mean fits better than subtracting 50, which would be half of set length
            adjustCL_split = (CLine_split-np.mean(CLine_split))+np.repeat(XY_split.reshape(XY_split.shape[0],1, XY_split.shape[1]), 
                                                                          CLine_split.shape[1], axis=1)
            
            ### feature calculation #####################################################################################
            ### calculates the vectors, their length and their inbetween angles of all centerline coordinates
            length, _, CLArctan = angle(CLine_split[:,::np.ceil(CLine_split.shape[1]/34).astype(int)])
            ### calculates the overall summed length of the pharynx
            SumLen = np.sum(length, axis=1)
            ### calculates the curvature of the pharynx
            Curvature = TotalAbsoluteCurvature(CLArctan, length)


            ### vectors and angle between nosetip, defined as first 5 CLine_split points, and center of mass
            _, tip2cm_arccos, _ = AngleLen(adjustCL_split, XY_split, hypotenuse = "v1", over="space", diffindex=[5,0])
            _, tip2tip_arccos, _ =  AngleLen(adjustCL_split[:,0], hypotenuse = "v1", over="frames")
            _, tip2tipMv_arccos, _ = AngleLen(adjustCL_split, adjustCL_split[:,0], hypotenuse = "v1", over="space", diffindex=[5,0])
            #angspeed_nose_sec = al.angular_vel_dt(tip2tip_arccos, dt=1)
            
            ### calculate reversal, as over 120deg 
            reversal_bin = np.where(tip2cm_arccos >= np.deg2rad(120), 1, 0)
            reversal_events = np.clip(np.diff(reversal_bin, axis=0), 0, 1)
            reversal_fract = pd.Series(reversal_bin.squeeze()).rolling(30, center=True).apply(lambda w: np.mean(w))
            reversal_rate = pd.Series(reversal_events.squeeze()).rolling(30, center=True).apply(lambda w: np.mean(w)*30)


            ### reshapes all features to fit the original
            Curvature, tip2cm_arccos, tip2tip_arccos, tip2tipMv_arccos, reversal_rate, reversal_fract = extendtooriginal((Curvature, tip2cm_arccos, tip2tip_arccos, tip2tipMv_arccos, 
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
                lowpass_d = lowpassfilter(PG_new[col].fillna(0).values/PG_new[col].mean(), 0.01)
                lowpass_toolarge = PG_new.shape[0]-lowpass_d.shape[0]
                if lowpass_toolarge < 0:
                    lowpass_d = lowpass_d[:lowpass_toolarge] 

                coefficients, frequencies = cwt_signal(lowpass_d, scales)#rec[:-1]
                maxfreq_idx = np.argmax(abs(coefficients), axis=0)
                maxfreq = maxfreq_idx.copy().astype('float')
                for i, f in enumerate(frequencies):
                    np.put(maxfreq, np.where(maxfreq_idx == i), [f])

                cols_coeff = pd.DataFrame(np.stack((abs(coefficients)), axis=1), columns=[f'{col}_cwt%02d'% scl for scl in scales])
                maxfreq_df = pd.DataFrame(maxfreq, columns=[f'{col}_maxfreq'])

                PG_new = pd.concat([PG_new, cols_coeff, maxfreq_df], axis=1)


            ### encode angular columns as cos sin
            deg_cols = PG_new.filter(regex='arctan$|arccos$').columns
            cos_ = encode_cos(PG_new[deg_cols]).rename(columns = lambda s: s.replace(s, s.split('_')[0]+'_cos'))
            sin_ = encode_sin(PG_new[deg_cols]).rename(columns = lambda s: s.replace(s, s.split('_')[0]+'_sin'))
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
        
        ### concat df splits, reindex to include nan in areas not present in PG_splits indices
        PG_joined = pd.concat(PG_splits)
        correct_idx = PG_joined.index
        PG_joined = PG_joined.reindex(pd.Index(range(0,len(PG))))
        
        ### postpad the end
        CL_joined = np.vstack([CLine_splits, np.full((len(CLine) - off, *CLine_splits.shape[1:]), np.nan)])
        XY_joined = np.vstack([XY_splits, np.full((len(PG) - off, *XY_splits.shape[1:]), np.nan)])

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
        XYs[fn] = XY_joined
        CLines[fn] = CL_joined
    return outs, XYs, CLines
    
def run(inpath, outpath, logger, return_XYCLine = False, skip_already = False, fps=30):
    if isinstance(inpath,str):
        inpath = [inpath]
    if isinstance(outpath,str):
        outpath = [outpath]
    ins, outs = setup(inpath, outpath, skip_already)
    outs, XYs, CLines = FeatureEngine(ins, outs, logger, fps=30)
    if return_XYCLine:
        return XYs, CLines
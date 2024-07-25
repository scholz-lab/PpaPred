# to avoid cluttering: new functionsfile (many functions in functionsPCA are depricated)
# contains less functions, more CPU friendly
import numpy as np
import pandas as pd
import logging
from scipy import ndimage
from scipy.signal import find_peaks
from functions.algebra import unitvector, unitvector_space, angle2vec

### IMPORT ###
""" will be depricated for json load, work-around for now
    iterating through Centerline of each frame
    writing Centerline data to numpy array
    adding centerline arrays to dict"""
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



def repro(angs, vlen, forPC = False):
    # reproject angles into 2D space
        # angs: np array with shape (frames, spine)
        # vlen: np array with shape (frames, spine, 1)
        # forPC: bool, True: for reprojection length is averaged
    # returns np array with x, y coordinates with shape (frames, spine, 2)
    lstCor = np.zeros((angs.shape[0], angs.shape[1]+2, 2))
    cumalpha = np.cumsum(angs, axis=1)
    if forPC:
        vlen = np.mean(vlen, axis=0)
        lstCor[:,1,0] = vlen[0]
        stepX = (np.cos(cumalpha[:,:,np.newaxis])*vlen[1:]).squeeze()
        stepY = (np.sin(cumalpha[:,:,np.newaxis])*vlen[1:]).squeeze()
        
    else:
        lstCor[:,1,0] = vlen[:,0].flatten()
        stepX = np.cos(cumalpha)*(vlen[:,1:].squeeze())
        stepY = np.sin(cumalpha)*(vlen[:,1:].squeeze())
        
    lstCor[:,2:,0] = stepX
    lstCor[:,2:,1] = stepY
    lstCor = np.cumsum(lstCor, axis=1)
    
    return lstCor


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

def orthlen2vec(coorDf1, coorDf2):
    # input: coorDf1 and coorDf2 with shape:x,x,2
    # returns length of orthogonal between mean vector (at axis 1) in coorDf1 and deviated vector in coorDf2 (presumed nose)
    # also returns angles between both those vectors
    
    # preparartion: calculate difference, length, unitarize
    c_vec = np.diff(np.mean(coorDf1, axis=1), axis=0) #old: coorDf1[:,0]
    c_len = np.linalg.norm(c_vec, axis = 1).reshape(c_vec.shape[0],1)
    c_unit_vec = np.divide(c_vec,c_len)
    b_vec = np.diff(np.mean(coorDf2, axis=1), axis=0)
    b_len = np.linalg.norm(b_vec, axis = 1).reshape(b_vec.shape[0],1)
    b_unit_vec = np.divide(b_vec,b_len)

    angles = np.full((max(len(coorDf1), len(coorDf2)),1),np.nan)
    a_len = np.full((max(len(coorDf1), len(coorDf2)),1),np.nan)
    last = 0
    for i in range(min(len(b_len), len(c_len))):
        # uses arctan to calculate angle of mean (Df1) and framewise vector (Df2) from origin (1,1), subtracted angles give angle between
        alpha = np.arctan2(c_unit_vec[i,1],c_unit_vec[i,0])-np.arctan2(b_unit_vec[i,1],b_unit_vec[i,0])
        if alpha > np.pi or alpha < -np.pi:
            while alpha < last-np.pi: alpha += 2*np.pi #check for out of range pi to -pi
            while alpha > last+np.pi: alpha -= 2*np.pi
        last = alpha
        angles[i] = alpha
        # uses sinus to calculate length of a, orthogonal nose deviance, based on length of framewise vector and alpha
        a_len[i] = np.sin(alpha)*c_len[i]
        """
                __a_ (orthoganal)
               |   /
        c      |  / b (framewise)
        (mean) |α/
               |/
        """
    
    return a_len, angles

def FreqAmpHeight(data, col, window = 30):
    ts_data = data[col]
    peaks,peak_meta = find_peaks(ts_data, prominence=np.std(ts_data), height=ndimage.uniform_filter1d(np.nan_to_num(ts_data), size=window*2))
    peaks_only = np.full(len(ts_data),np.nan)
    peaks_only[peaks] = ts_data[peaks]
    promin_only = np.zeros(len(ts_data))
    promin_only[peaks] = peak_meta['prominences']
    height_only = np.zeros(len(ts_data))
    height_only[peaks] = peak_meta['peak_heights']
    freq = pd.Series(peaks_only).rolling(window).count()
    amp = ndimage.maximum_filter(promin_only, size=window) # uniform_filter1d or maximum_filter?
    height = ndimage.maximum_filter(height_only, size=window)
    
    return peaks, freq, amp, height


def block_avg(df, N=30, return_blown=True):
    groups_of_N = pd.Series(np.repeat(range(int(np.ceil(len(df)/N))), N))
    block_avg = df.groupby(groups_of_N).median()
    if return_blown:
        blown_avg = block_avg.loc[block_avg.index.repeat(N)].reset_index(drop=True)
        if len(blown_avg) > len(df):
            overhang = len(blown_avg)-len(df)
            blown_avg.drop(blown_avg.tail(overhang).index,inplace = True)

        return blown_avg
    else:
        return block_avg
    
def block_most(df, N=30, return_blown=True):
    groups_of_N = pd.Series(np.repeat(range(int(np.ceil(len(df)/N))), N))
    block_most = df.groupby(groups_of_N).agg(pd.Series.mode)
    for col in block_most:
        block_most[col] = block_most[col].apply(lambda x: x[0] if x.shape >= (1,) else x) # gets the first value if values have same frequency
    if return_blown:
        blown_most = block_most.loc[block_most.index.repeat(N)].reset_index(drop=True)
        if len(blown_most) > len(df):
            overhang = len(blown_most)-len(df)
            blown_most.drop(blown_most.tail(overhang).index,inplace = True)
        return blown_most
    else:
        return block_most

def onoff_dict(arr_raw, labels = range(-1,4), return_duration=False, return_transitions = False, return_all = False):
    if isinstance(arr_raw, pd.Series) or isinstance(arr_raw, pd.DataFrame):
        arr_raw = arr_raw.values
    arr_s = arr_raw[1:]
    arr = arr_raw[:-1]

    arr_transi =  np.append(arr[arr != arr_s], arr[-1])
    arr_onset = np.concatenate([[0],np.where([arr != arr_s])[1]+1])
    arr_onnext= np.append(np.array((arr_onset)[1:]), [len(arr)+1])
    arr_dur = (arr_onnext)-arr_onset
    arr_onoff = {}

    for b in labels:
        b_idx = np.where(arr_transi == b)
        b_onoff = list(zip(arr_onset[b_idx], arr_dur[b_idx]))
        arr_onoff[b] = b_onoff

    if return_all == True:
        return arr_onoff, arr_dur, arr_transi, arr_onset, arr_onnext
    elif return_transitions == True and return_duration == True:
        return arr_onoff, arr_dur, arr_transi
    elif return_duration == True or return_transitions == True:
        if return_duration == True:
            return arr_onoff, arr_dur
        if return_transitions == True:
            return arr_onoff, arr_transi
    else:
        return arr_onoff
    
def onoff_dict(arr_raw, labels = range(-1,6), return_duration=False, return_transitions = False, return_all = False, treatasone=True):
    #TODO the variable transi is better named sequence
    if not isinstance(arr_raw, list):
        arr_raw = [arr_raw]

    arr_onoff = {}
    arr_transi =  []
    arr_onset =  []
    arr_onnext =  []
    arr_dur =  []
    total_dur = 0
    for i,a in enumerate(arr_raw):
        if isinstance(a, pd.Series) or isinstance(a, pd.DataFrame):
            a = a.values
        arr_s = a[1:]
        arr = a[:-1]

        bool_diff = arr != arr_s
        transi = np.append(arr[bool_diff], arr[-1])
        onset = np.append([0],np.where(bool_diff)[0]+1)
        onnext = np.append(np.array((onset)[1:]), [len(arr)+1])
        dur = onnext-onset
        arr_transi.append(transi)
        arr_onset.append(onset)
        arr_onnext.append(onnext)
        arr_dur.append(dur)

        if treatasone:
            if i > 0:
                total_dur += arr_onnext[i-1][-1]

        for b in np.unique(a):
            b_idx = transi == b
            b_onoff = list(zip(onset[b_idx]+total_dur, dur[b_idx]))
            if b in arr_onoff.keys():
                arr_onoff[b] = arr_onoff[b]+b_onoff
            else:
                arr_onoff[b] = b_onoff
    if treatasone:
        arr_dur = np.concatenate(arr_dur)
        arr_transi = np.concatenate(arr_transi)
        arr_onset = np.concatenate([a+arr_onnext[i-1][-1] if i > 0 else a for i,a in enumerate(arr_onset)])
        arr_onnext = np.concatenate([a+arr_onnext[i-1][-1] if i > 0 else a for i,a in enumerate(arr_onnext)])
        # might hav to work here on further, change how arr_onset and arr_onnext are daved

    if return_all == True:
        return arr_onoff, arr_dur, arr_transi, arr_onset, arr_onnext
    elif return_transitions == True and return_duration == True:
        return arr_onoff, arr_dur, arr_transi
    elif return_duration == True or return_transitions == True:
        if return_duration == True:
            return arr_onoff, arr_dur
        if return_transitions == True:
            return arr_onoff, arr_transi
    else:
        return arr_onoff
    
def ffill_bfill(arr, size):
    if isinstance(arr, pd.Series) or isinstance(arr, pd.DataFrame):
        arr = arr.values
    arr_filled = arr.copy()
    for size in range(2, size):
        bsize = size//2
        fsize = size - bsize
        arr_onoff, arr_dur, arr_transi, arr_on, arr_off = onoff_dict(arr_filled, np.unique(arr_filled), return_all = True)
        if not any(arr_dur < size):
            continue
        else:
            start = arr_on[arr_dur < size]
            stop = arr_off[arr_dur < size]
            st_st = list(zip(start,stop))
            arr_nan = arr_filled.copy().astype(float)
            arr_nan[np.hstack([np.arange(a, o) for a, o in st_st])] = np.nan
            arr_filled = pd.DataFrame(arr_nan).fillna(method="pad", limit=fsize).fillna(method="bfill", limit=bsize).fillna(method="pad").values.flatten()
    return arr_filled

def safe_round_matrix(arr, axis = 1, decimals = 2):
    arr = arr.copy()
    if axis == 0:
        arr = arr.T
    # round to desired decimals
    arr_round = np.round(arr, decimals)
    # find rows where sum is not retained
    not_retained_sum, = np.where(arr_round.sum(axis=1) != arr.sum(axis=1))
    
    for idx in not_retained_sum:
        # calculate the value that is missing to reach original
        missing_value = [np.round(arr[idx].sum() - arr_round[idx].sum(),decimals)]
        print(missing_value)
        # if missing_value is larger than the rounding step, defined by decimals, split missing value up
        if missing_value[0] > 1*10**-decimals:
            missing_value = [1*10**-decimals]*missing_value[0]
        
        # get index where it would be fairest to round up/down
        if missing_value[0] >= 0:
            best_remainder = np.argsort(arr[idx] - arr_round[idx])[::-1] # invert so that highest remainder is first
        else:
            best_remainder = np.argsort(arr[idx] - arr_round[idx]) # do not invert so that lowest remainder is first
        
        # add missing values to first indices in best_remainder
        for i in range(len(missing_value)):
            arr[idx, best_remainder[i]] += missing_value[i]
    
    # with added missing values, round again
    arr_round_safe = np.round(arr,decimals)
    if axis == 0:
        arr_round_safe = arr_round_safe.T
    return arr_round_safe
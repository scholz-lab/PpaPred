# to avoid cluttering: new functionsfile (many functions in functionsPCA are depricated)
# contains less functions, more CPU friendly
import numpy as np
import pandas as pd
import pywt
from functions.algebra import unitvector, unitvector_space, angle2vec

### IMPORT ###


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

def velocity(x,y, scaling, fps, dt=5):
    """
    Calculates the velocity from euclidean distance given the x and y coordinates over time.
    Parameters:
        x, y : numpy arrays representing the x and y coordinates of the object over time.
        scaling : a scaling factor to convert the units of x and y coordinates.
        fps : frames per second, the rate at which the coordinates are sampled.
        dt : time interval over which the velocity is calculated. Default is 5.
    Returns:
        dist*fps : numpy array representing the velocity of the object over time."""
    x, y = x*scaling, y*scaling
    dist = np.sqrt(x.diff(periods=dt)**2 + y.diff(periods=dt)**2)/dt
    return dist*fps

# used
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

# used
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

# used
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


# used
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
        if missing_value[0] > 10**-decimals:
            missing_value = [10**-decimals] * int(missing_value[0]/(10**-decimals))
        
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
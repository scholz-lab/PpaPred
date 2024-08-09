import numpy as np
from scipy import stats

def extendtooriginal(arrays, org_shape):
    extended = []
    for arr in arrays:
        nans = np.full(org_shape, np.nan)
        nans.flat[:arr.size] = arr
        extended.append(nans)
    return extended

def unitvector(xyarray, diff_step = 1):
    """
    Computes unit vectors and vector lengths from a 2D array of coordinates.
    
    Args:
        xyarray (np.ndarray): Input array of 2D coordinates (shape: [n_points, 2]).
    
    Returns:
        tuple: A tuple containing:
            - unit_vec (np.ndarray): Unit vectors corresponding to each coordinate.
            - vlen (np.ndarray): Vector lengths for each coordinate.
            - base (np.ndarray): Differences between consecutive coordinates.
    """
    base = xyarray[diff_step:] - xyarray[:-diff_step]
    vlen = np.linalg.norm(base, axis = 1).reshape(base.shape[0],1)
    unit_vec = np.divide(base,vlen)
    unit_vec = np.nan_to_num(unit_vec)
    return unit_vec, vlen/diff_step, base

def unitvector_space(xyarray, diffindex=[0,-1]):
    """
    Computes unit vectors and vector lengths for a given array of 2D points at a defined index.

    Args:
        xyarray (numpy.ndarray): A 2D array containing point coordinates.
        diffindex (list, optional): Indices for points to compute base vector from (default: [0, -1]).

    Returns:
        numpy.ndarray: Array of unit vectors.
        numpy.ndarray: Array of vector lengths.
        numpy.ndarray: Base vectors.
    """
    base = xyarray[:,diffindex[1]]-xyarray[:,diffindex[0]]
    vlen = np.linalg.norm(base, axis = 1).reshape(base.shape[0],1)
    unit_vec = np.divide(base,vlen)
    unit_vec = np.nan_to_num(unit_vec)
    return unit_vec, vlen, base


def AngleLen (v1, v2=None, hypotenuse = "v1", over="frames", v2_over = 'frames', v1_args={}, v2_args={}, v2_diff = 1, angletype=np.arccos):
    """
    Computes the length of the difference vector and the angle between two vectors.

    Args:
        v1 (numpy.ndarray): First vector (with 2D coordinates).
        v2 (numpy.ndarray, optional): Second vector (default: None).
        hypotenuse (str, optional): Determines which vector's length to use as the hypotenuse (default: "v1").
        over (str, optional): Specifies whether to compute over "frames" or "space" (default: "frames").
        **args: Additional arguments for unit vector computation, only applicable if over = "space".

    Returns:
        numpy.ndarray: Length of the difference vector.
        numpy.ndarray: Angle (in radians) between v1 and v2.
        numpy.ndarray: Difference vector (base vector).
    """
    func_dict = {'frames': unitvector,
                 'space': unitvector_space}
    
    v1_unitfunc = func_dict[over]
    v2_unitfunc = func_dict[v2_over]  
    v1_unit, v1_len, v1_diff = v1_unitfunc(v1,**v1_args)
    if not v2 is None:
        v2_unit, v2_len, v1_diff = v2_unitfunc(v2, **v2_args)
    else:
        v2_unit, v2_len, v1_diff = v1_unit[v2_diff:], v1_len[v2_diff:], v1_diff[v2_diff:]
    
    hyp = {"v1":v1_len, "v2":v2_len}
    hypo_len = hyp[hypotenuse]
    adjecent_len = hyp[[k for k in hyp.keys() if k != hypotenuse][0]]
    
    crop = min(len(v1_unit), len(v2_unit))
    
    dotProduct = v1_unit[:crop,0]*v2_unit[:crop,0] +v1_unit[:crop,1]*v2_unit[:crop,1]
    arccos = angletype(dotProduct) # mod of Vector is 1, so /mod can be left away  #arccos
    
    # calculate length of opposite of calculated angle: diveregence from v2 (cm) vector
    opposite_len = np.multiply(hypo_len[:crop].flatten(), abs(np.sin(arccos[:crop]))) 
    
    return opposite_len, arccos, v1_diff

def AngleLenArcTan (v1, v2=None, hypotenuse = "v1", over="frames",**args):
    if over == "frames":
        unitfunction = ak.unitvector
    elif over == "space":
        unitfunction = al.unitvector_space
        
    v1_unit, v1_len, v1_diff = unitfunction(v1,**args)
    if not v2 is None:
        v2_unit, v2_len, v1_diff = al.unitvector(v2)
    else:
        v2_unit, v2_len, v1_diff = v1_unit[1:], v1_len[1:], v1_diff[1:]
    
    hyp = {"v1":v1_len, "v2":v2_len}
    hyplen = hyp[hypotenuse]
    print(v1_unit, v2_unit)
    crop = min(len(v1_unit), len(v2_unit))
    #x1, y1, x2, y2 = v1_unit[:crop,0], v1_unit[:crop,1], v2_unit[:crop,0], v2_unit[:crop,1]
    dotProduct = v1_unit[:crop,0]*v2_unit[:crop,0] +v1_unit[:crop,1]*v2_unit[:crop,1]
    arctan = np.arctan2(np.cross(v1_unit[:crop],v2_unit[:crop]),dotProduct)
    #arcsin = np.arcsin(dotProduct)
    
    difflen = np.multiply(np.sin(arctan[:crop]).flatten(),hyplen[:crop].flatten())
    
    return difflen, arctan, v1_diff

def TotalAbsoluteCurvature(Phi_i, L_i, axis=1):
    """
    Calcultes total absolute curvature of discrete curves, given the vector angle by Phi_i and the vector length by L_i
    """
    A_i= ((L_i[:,:-1]+L_i[:,1:])/2)
    k_i = Phi_i.squeeze()/A_i.squeeze()
    k = np.sum(abs(k_i), axis=axis)
    return k

'''
def LineAngle(Cl):
    """Use centerline points to calculate relative angles.
    Cl = (Nsamples, 100, 2) array
    """
    # vector components for relative vectors
    vec = np.diff(Cl, axis = 1)
    # normalize vectors. calculates the length of vectors
    length = np.linalg.norm(vec, axis = 2).reshape(vec.shape[0],vec.shape[1],1)
    unit_vec = vec/length
    # define vectors to calculate angles from: shift
    v1_unit = unit_vec[:,:-1] # from 0 to forelast
    v2_unit = unit_vec[:,1:] # from 1 to last
    # angles
    dotProduct = v1_unit[:,:,0]*v2_unit[:,:,0] +v1_unit[:,:,1]*v2_unit[:,:,1]
    crossProduct = np.cross(v1_unit,v2_unit)

    arccos = np.arccos(dotProduct) # mod of Vector is 1, so /mod can be left away  #arccos
    arcsin = np.arcsin(crossProduct)
            
    return length,vec,arccos,arcsin'''


def angular_vel_dt(arr, dt=1, fps=30):
    return stats.circmean(np.lib.stride_tricks.sliding_window_view(arr, int(dt*fps)), 
                          #high=np.pi/2,low=-np.pi/2,
                          high=np.pi,low=-np.pi,
                          axis=1)*(fps*dt)

def angle2vec(nose_unit_frames, nose_unit_space):
    crop = min(len(nose_unit_frames), len(nose_unit_space))
    dot_noses = nose_unit_frames[:crop,0]*nose_unit_space[:crop,0] +nose_unit_frames[:crop,1]*nose_unit_space[:crop,1]
    angle_noses = np.arccos(dot_noses) # mod of Vector is 1, so /mod can be left away
    return angle_noses

def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
    vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    xy = np.c_[[length, 0], [0, 0], vec2*length].T + np.array(pos)
    ax.plot(*xy.T, color=acol)
    return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

# encode cos, sin
def encode_cos(x):
    return np.cos(x)
def encode_sin(x):
    return np.sin(x)


###### from https://matplotlib.org/stable/gallery/text_labels_and_annotations/angle_annotation.html#sphx-glr-gallery-text-labels-and-annotations-angle-annotation-py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

        
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition
        
        if self.check_180:
            self.vec1 = p2
            self.vec2 = p1

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy
        
    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass
    
    def check_180(self):
        if self.get_theta(self.vec1) + self.get_theta(self.vec2) > 90:
            return True
        else: 
            return False


    # Redefine attributes of the Arc to always give values in pixel space

    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])
            

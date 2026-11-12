#!/usr/bin/env python3
"""
QCar V23 scanline tracker + Gazebo white-boundary fallback.

Design goal
-----------
Follow the behaviour visible in the user's reference screenshots:
- LEFT boundary: actual yellow painted line.
- RIGHT boundary: actual dark-road -> bright/white curb transition.
- Boundaries are tracked row-by-row as real image points.
- No global boundary polynomial is extrapolated across unsupported image rows.
- Everything left of the yellow boundary is ignored for right-boundary search.
- Centreline is the point-by-point midpoint where both sides are visible.
- Metric CTE/heading/curvature are calculated from the sampled centreline.

ROS input:
    /front_camera/image_raw       sensor_msgs/Image
    /front_camera/camera_info     sensor_msgs/CameraInfo (optional)
    /state/velocity               std_msgs/Float64 (optional/debug only)

ROS output:
    /control/cte                  std_msgs/Float64 [m]
    /control/heading_error        std_msgs/Float64 [rad]
    /control/curvature            std_msgs/Float64 [1/m]
    /lane_detection/status        std_msgs/Float64
        0 LOST
        2 PARTIAL
        3 FULL
    /lane_detection/debug_image
    /lane_detection/yellow_mask
    /lane_detection/right_edge_mask
    /lane_detection/corridor_mask
    /lane_detection/reference_path    nav_msgs/Path [odom frame] (new for rcognita)

Sign convention:
    cte > 0              lane centre is left of vehicle
    heading_error > 0    lane tangent points vehicle/camera right
    curvature > 0        lane bends left
"""

import math
import time
import json
from collections import deque
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from nav_msgs.msg import Odometry, Path as ROSPath
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String


LANE_STATUS_LOST = 0.0
LANE_STATUS_PARTIAL = 2.0
LANE_STATUS_FULL = 3.0


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def camera_centerline_to_world(forward, lateral, base_x, base_y, base_yaw,
                               camera_x, camera_y, camera_yaw=0.0):
    """Convert V23 camera-aligned metric centerline to the odometry XY frame.

    Camera coordinates are X forward and Y left, NOT optical-frame pixels.
    ``base_x,base_y,base_yaw`` must locate the Xacro ``base`` link (not the
    rear axle). Camera translation is expressed in that base link's frame.
    """
    points = np.column_stack([np.asarray(forward, dtype=float),
                              np.asarray(lateral, dtype=float)])
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError('Nonfinite camera centerline')
    ca, sa = math.cos(base_yaw), math.sin(base_yaw)
    camera_origin = np.array([base_x + ca * camera_x - sa * camera_y,
                              base_y + sa * camera_x + ca * camera_y])
    angle = base_yaw + camera_yaw
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rotation.T + camera_origin



def sanitize_metric_lane_path(points, spacing_m=0.05, fit_radius_m=0.13,
                              min_span_m=0.28, max_curvature_m_inv=2.5,
                              max_observation_gap_m=0.25):
    """Smooth/resample ONLY the observed metric camera centerline, no extrapolation.

    Inputs are [forward_m, left_m] near-to-far samples from V23's two measured
    lane boundaries. Independent variable is forward displacement: a local
    *robust weighted linear fit* attenuates millimetre-scale depth jitter,
    then the observed interval is resampled at approximately uniform spacing.
    Neither lane boundaries nor the controller's geometric error code change.

    Return (cleaned Nx2 array, diagnostics) or (None, diagnostics) when there
    is too little supported geometry / a bend too sharp to safely interpret.
    """
    info = {'valid': False, 'reason': '', 'span_m': 0., 'p95_curvature_m_inv': float('nan')}
    raw = np.asarray(points, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 6 or not np.isfinite(raw).all():
        info['reason'] = 'invalid metric centerline samples'
        return None, info
    if (spacing_m <= 0 or fit_radius_m <= 0 or min_span_m <= 0 or
            max_curvature_m_inv <= 0 or max_observation_gap_m <= 0):
        raise ValueError('Lane geometry parameters must be positive')
    # The camera measurement is ordered near-to-far, but width-derived depth
    # may contain small nonmonotonic/duplicate forward samples. Drop them.
    x, y = raw[:, 0], raw[:, 1]
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    keep = np.r_[True, np.diff(x) > 0.002]
    x, y = x[keep], y[keep]
    if len(x) < 6 or x[-1] - x[0] < min_span_m:
        info['reason'] = 'observed lane has insufficient forward span'
        return None, info
    info['span_m'] = float(x[-1] - x[0])
    # Don't interpolate through a long camera blind spot / missing lane section.
    if float(np.max(np.diff(x))) > max_observation_gap_m:
        info['reason'] = 'gap between observed lane samples is too large'
        return None, info
    count = min(85, max(2, int(np.ceil(info['span_m'] / spacing_m)) + 1))
    grid = np.linspace(x[0], x[-1], count)
    smoothed = np.empty(count, dtype=float)
    for i, xi in enumerate(grid):
        within = np.abs(x - xi) <= fit_radius_m
        if np.count_nonzero(within) < 3:
            neighbours = np.argsort(np.abs(x - xi))[:min(5, len(x))]
            within = np.zeros(len(x), dtype=bool)
            within[neighbours] = True
        dx, ly = x[within] - xi, y[within]
        A = np.column_stack((np.ones(len(dx)), dx))
        base_weights = 1. / (1. + (dx / 0.075)**2)
        weights = base_weights.copy()
        for _ in range(3):
            w = np.sqrt(weights)
            co = np.linalg.lstsq(A*w[:, None], ly*w, rcond=None)[0]
            residual = ly - A@co
            robust_scale = max(0.008, 1.4826 * np.median(np.abs(residual - np.median(residual))))
            weights = base_weights * np.minimum(1., (1.8 * robust_scale) / (np.abs(residual) + 1.e-9))
        smoothed[i] = float(co[0])
    path = np.column_stack((grid, smoothed))
    d = np.diff(path, axis=0)
    seg_len = np.linalg.norm(d, axis=1)
    headings = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    curvature = np.abs(np.diff(headings) / np.maximum(0.5*(seg_len[1:] + seg_len[:-1]), 1.e-8))
    p95 = float(np.percentile(curvature, 95)) if len(curvature) else 0.
    info['p95_curvature_m_inv'] = p95
    info['points'] = len(path)
    if not np.isfinite(p95) or p95 > max_curvature_m_inv:
        info['reason'] = 'observed path curvature exceeds configured safety bound'
        return None, info
    info['valid'] = True
    return path, info

def remember_yellow_center_offset(centre_points, left_points, *, min_points=8, img_width=820):
    """Learn image-row-specific centre minus yellow from an observed FULL frame.

    Never uses guessed widths. Only uses pixels matched from genuine TWO-edge
    centreline and current yellow observations. Returns ascending [row,offset].
    """
    if not centre_points or not left_points:
        return None
    left = np.asarray(left_points, dtype=float)
    if left.ndim != 2 or left.shape[1] != 2 or len(left) < 8 or not np.isfinite(left).all():
        return None
    left = left[np.argsort(left[:, 1])]
    left_y, unique_indices = np.unique(left[:, 1], return_index=True)
    left_x = left[unique_indices, 0]
    if len(left_y) < 8:
        return None
    offsets = []
    for centre in centre_points:
        if len(centre) != 3:
            continue
        cx, y, measured_width = map(float, centre)
        if not (np.isfinite(cx) and np.isfinite(y) and np.isfinite(measured_width)):
            continue
        if y < left_y[0] or y > left_y[-1]:
            continue
        lx = float(np.interp(y, left_y, left_x))
        delta = cx - lx
        if 6.0 < delta < 0.48 * img_width and abs(2.0*delta - measured_width) <= max(12.,0.2*measured_width):
            offsets.append((y, delta))
    if len(offsets) < min_points:
        return None
    arr = np.asarray(sorted(offsets), dtype=float)
    if arr[-1, 0] - arr[0, 0] < 25.0:
        return None
    return arr


def centre_from_visible_yellow(left_points, offset_profile, *, min_points=12,
                               min_vertical_span_px=35.0, img_width=820):
    """Offset only CURRENT observed yellow rows by LAST FULL measured offset.

    No fabricated yellow points, no image-row extrapolation, no change to FULL
    midpoint logic. Offset changes with image row to preserve perspective.
    """
    if offset_profile is None or not left_points:
        return None
    offset = np.asarray(offset_profile, dtype=float)
    left = np.asarray(left_points, dtype=float)
    if offset.ndim != 2 or offset.shape[1] != 2 or left.ndim != 2 or left.shape[1] != 2:
        return None
    if not np.isfinite(offset).all() or not np.isfinite(left).all():
        return None
    if len(offset) < 8 or len(left) < min_points:
        return None
    left = left[np.argsort(-left[:, 1])]  # near-to-far
    y0, y1 = float(offset[0, 0]), float(offset[-1, 0])
    selected = left[(left[:, 1] >= y0) & (left[:, 1] <= y1)]
    if len(selected) < min_points or float(np.ptp(selected[:,1])) < min_vertical_span_px:
        return None
    # Reduce the influence of spikes on the remaining yellow boundary.
    if len(selected) >= 8:
        x = selected[:, 0]
        filtered = np.asarray([np.median(x[max(i-2,0):min(i+3,len(x))])
                               for i in range(len(x))], dtype=float)
        if float(np.percentile(np.abs(x-filtered), 95)) > 15.0:
            return None
    result = []
    for x,y in selected:
        half_width = float(np.interp(y,offset[:,0],offset[:,1]))
        cx = float(x+half_width)
        if not 0.0 <= cx < img_width:
            return None
        result.append((cx,float(y),2.0*half_width))
    return result if len(result) >= min_points else None





def learned_ground_scale_profile(centre_points, *, min_samples=12):
    """FULL-lane row->pixel-width calibration from *observed* two-edge pairs.

    This is a locally learned perspective scale (NOT a measured camera-to-ground
    homography).  It is trustworthy only while the camera mounting and physical
    lane width remain unchanged. In decreasing image row the visible lane width
    should not grow. A bounded median/isotonic fit removes width jitter that
    otherwise explodes when differentiated into curvature.
    """
    arr = np.asarray(centre_points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < min_samples or not np.isfinite(arr).all():
        return None
    arr = arr[np.argsort(arr[:,1])]
    y, widths = np.unique(arr[:,1],return_index=True)
    w = arr[widths,2]
    if len(y) < min_samples or float(y[-1]-y[0]) < 40.0 or np.any(w <= 15.0):
        return None
    wmed = median_filter_1d(w, radius=2)
    # Robust monotonic projection: width should increase toward the camera.
    # Do not fabricate observations beyond the FULL lane's image-row support.
    wmono = np.maximum.accumulate(wmed)
    if wmono[-1] < wmono[0] * 1.10:
        return None
    if float(np.percentile(abs(w-wmono),90)) > max(18.,0.20*float(np.median(w))):
        return None
    return np.column_stack((y,wmono))


def metric_yellow_from_ground_scale(left_points, profile, *, img_width, fx,
                                    lane_width_m, min_points=12,
                                    min_row_span_px=35.0, max_yellow_spike_px=18.0):
    """Project LIVE yellow pixels with a scale learned during a real FULL lane.

    Ground proxy is (forward, left) in metres using Z(y)=fx*W/width(y),
    lateral=(W/2 - yellow_x - width(y)/2)*W/width(y). Both width(y) and Z(y)
    are retained from FULL detection and thus cannot be altered by a noisy
    single-boundary pseudo-width on the next frame. No unsupported image rows
    are extrapolated. Returns camera metric samples + recovery debug pixels.
    """
    left=np.asarray(left_points,float)
    prof=np.asarray(profile,float) if profile is not None else np.empty((0,2))
    if (left.ndim!=2 or left.shape[1]!=2 or prof.ndim!=2 or prof.shape[1]!=2
            or len(left)<min_points or len(prof)<min_points
            or not np.isfinite(left).all() or not np.isfinite(prof).all()
            or fx<=1 or lane_width_m<=0 or img_width<20):
        return None,None,{'reason':'invalid or missing full ground-scale calibration'}
    left=left[np.argsort(-left[:,1])]
    inside=(left[:,1]>=prof[0,0]) & (left[:,1]<=prof[-1,0])
    left=left[inside]
    if len(left)<min_points or np.ptp(left[:,1])<min_row_span_px:
        return None,None,{'reason':'yellow visible rows do not cover learned FULL scale'}
    ys=left[:,1]
    x=left[:,0]
    xs=median_filter_1d(x,radius=2)
    if np.percentile(np.abs(x-xs),95)>max_yellow_spike_px:
        return None,None,{'reason':'yellow boundary has inconsistent lateral spikes'}
    # Reject long jumps even if there are many raw yellow pixels overall.
    if np.max(np.diff(-ys))>20.0:
        return None,None,{'reason':'yellow scanline support contains large gap'}
    w=np.interp(ys,prof[:,0],prof[:,1]); depth=fx*lane_width_m/w
    depth0=fx*lane_width_m/prof[-1,1]
    forward=depth-depth0
    lateral=(0.5*img_width-xs-0.5*w)*lane_width_m/w
    centre_x=xs+0.5*w
    valid=np.isfinite(forward)&np.isfinite(lateral)&(forward>=-1.e-6)&(centre_x>=0)&(centre_x<img_width)
    forward=np.maximum(0.,forward[valid]); lateral=lateral[valid]
    ys=ys[valid];centre_x=centre_x[valid];w=w[valid]
    if len(forward)<min_points:
        return None,None,{'reason':'insufficient supported yellow ground points'}
    use=np.r_[True,np.diff(forward)>0.002]
    points=np.column_stack((forward[use],lateral[use]))
    debug=[(float(cx),float(y),float(wi)) for cx,y,wi in zip(centre_x,ys,w)]
    if len(points)<8 or points[-1,0]-points[0,0]<0.20:
        return None,None,{'reason':'reconstructed yellow path has insufficient forward preview'}
    return points,debug,{'reason':'live yellow + FULL row-ground scale',
                         'raw_span_m':float(points[-1,0]-points[0,0])}



def project_yellow_pixels_to_ground(left_points, *, fx, fy, cx, cy, camera_height_m,
                                    camera_pitch_rad, max_range_m=1.40):
    """Intersect rectified pinhole-camera rays with a flat track (x forward, y left).

    Pitch is positive DOWN. Camera optical x points right, y down, z forward.
    Crucially, no distance or field-of-view calibration is learned from a FULL
    lane's changing pixel width; the camera mounting is treated as static.
    Do not run if camera pitch/height or ground flatness are incorrect.
    """
    info={'reason':'', 'pixels_used':0}
    pix=np.asarray(left_points,dtype=float)
    if (pix.ndim!=2 or pix.shape[1]!=2 or len(pix)<10 or not np.isfinite(pix).all()
            or min(fx,fy,camera_height_m)<=0 or max_range_m<=0):
        info['reason']='insufficient yellow or invalid calibrated ground projection'
        return None,None,info
    pix=pix[np.argsort(-pix[:,1])] # nearest image row first
    u,v=pix[:,0],pix[:,1]
    down=(v-cy)/fy
    den=math.sin(camera_pitch_rad)+down*math.cos(camera_pitch_rad)
    use=den>0.035  # avoid near-horizon ground intersections / unstable scales
    x=np.divide(camera_height_m*(math.cos(camera_pitch_rad)-down*math.sin(camera_pitch_rad)),
                den,out=np.full_like(den,np.nan),where=use)
    y=np.divide(-camera_height_m*(u-cx)/fx, den,out=np.full_like(den,np.nan),where=use)
    use &= np.isfinite(x)&np.isfinite(y)&(x>=0.06)&(x<=max_range_m)
    points=np.column_stack((x[use],y[use])); pixels=pix[use]
    if len(points)<12:
        info['reason']='too few visible yellow ground intersections'
        return None,None,info
    # Original scanline tracker is near-to-far. A tiny normal-depth jitter may
    # reorder duplicate rows; keep consecutive supported readings only.
    good=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>0.001]
    points=points[good];pixels=pixels[good]
    if len(points)<12:
        info['reason']='insufficient independent ground samples'
        return None,None,info
    jumps=np.linalg.norm(np.diff(points,axis=0),axis=1)
    big=np.flatnonzero(jumps>0.23)
    if len(big):
        # Truncate at the first non-physical gap; do not bridge it.
        stop=int(big[0]+1);points=points[:stop];pixels=pixels[:stop]
    if len(points)<12 or np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1))<0.20:
        info['reason']='observed yellow has insufficient continuous ground arc length'
        return None,None,info
    info.update(reason='calibrated height + provisional pitch ray-plane intersection',
                pixels_used=len(points))
    return points,pixels,info


def recover_ground_curve_normal_from_yellow(left_points, *, fx, fy, cx, cy,
                                          camera_height_m, camera_pitch_rad,
                                          lane_width_m, max_range_m=1.40,
                                          fit_max_rmse_m=0.055,
                                          min_inlier_fraction=0.75):
    """Construct a centreline from current yellow observations in ground XY.

    Select a robust line or circular arc in ground coordinates, then offset to
    the right by W/2 along its true 2D normal. No image-row overlap requirement;
    no synthesized path beyond the currently observed yellow support.
    """
    info={'reason':'','valid':False,'fit_model':'none',
          'fit_rmse_m':float('nan'),'inlier_fraction':0.0,'raw_span_m':0.0}
    points,pixels,projection=project_yellow_pixels_to_ground(
        left_points,fx=fx,fy=fy,cx=cx,cy=cy,camera_height_m=camera_height_m,
        camera_pitch_rad=camera_pitch_rad,max_range_m=max_range_m)
    if points is None:
        info['reason']=projection['reason'];return None,None,info
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<.20:
        info['reason']='yellow-only observed arc too short';return None,None,info
    # Robust model choice. Circle covers right and left curved lanes; line
    # covers straight segments. Work in arc length, never assume x increases.
    median=np.median(points,axis=0)
    _,evec=np.linalg.eigh((points-median).T@(points-median))
    tangent=evec[:,1]
    if np.dot(tangent,points[-1]-points[0])<0:tangent=-tangent
    dist=np.abs((points[:,0]-median[0])*tangent[1]-(points[:,1]-median[1])*tangent[0])
    inliers_line=dist<=max(.035,3*np.median(dist)+.012)
    if np.count_nonzero(inliers_line)>=10:
        base=np.mean(points[inliers_line],axis=0)
        _,v=np.linalg.eigh((points[inliers_line]-base).T@(points[inliers_line]-base))
        tangent=v[:,1]
        if np.dot(tangent,points[-1]-points[0])<0:tangent=-tangent
        median=base
    dist=np.abs((points[:,0]-median[0])*tangent[1]-(points[:,1]-median[1])*tangent[0])
    line_rms=float(np.sqrt(np.mean(np.sort(dist)[:max(10,int(.85*len(points)))]**2)))
    choice='line';model_arc=None;residual=dist
    n=len(points);indices=np.unique(np.linspace(1,n-2,min(n-2,13)).astype(int))
    trials=[]
    for ia in indices[:5]:
        for ib in indices[len(indices)//3:len(indices)//3+5]:
            for ic in indices[-5:]:
                if not (ia+3<=ib and ib+3<=ic):continue
                tri=points[[ia,ib,ic]]
                A=np.column_stack((2*tri[:,0],2*tri[:,1],np.ones(3)))
                if abs(np.linalg.det(A))<1.e-7:continue
                try:q=np.linalg.solve(A,np.sum(tri*tri,axis=1))
                except np.linalg.LinAlgError:continue
                cc=q[:2];radius=float(np.sqrt(max(0.,q[2]+cc@cc)))
                if .31<=radius<=8.0:
                    err=np.abs(np.linalg.norm(points-cc,axis=1)-radius)
                    trials.append((float(np.median(err)),cc,radius))
    if trials:
        _,cc,radius=min(trials,key=lambda t:t[0])
        for _ in range(3):
            err=np.abs(np.linalg.norm(points-cc,axis=1)-radius)
            mask=err<=max(.04,3*np.median(err)+.015)
            if np.count_nonzero(mask)<10:break
            pp=points[mask];A=np.column_stack((2*pp[:,0],2*pp[:,1],np.ones(len(pp))))
            q,*_=np.linalg.lstsq(A,np.sum(pp*pp,axis=1),rcond=None)
            cc=q[:2];radius=float(np.sqrt(max(0.,q[2]+cc@cc)))
        err=np.abs(np.linalg.norm(points-cc,axis=1)-radius)
        angles=np.unwrap(np.arctan2(points[:,1]-cc[1],points[:,0]-cc[0]))
        delta=float(angles[-1]-angles[0]);sweep=abs(delta)
        monotone=abs(delta)>1e-5 and np.mean(np.sign(np.diff(angles))==np.sign(delta))>.70
        circ_rms=float(np.sqrt(np.mean(np.sort(err)[:max(10,int(.85*n))]**2)))
        if (.31<=radius<=8.0 and .12<=sweep<=2.5 and monotone
                and circ_rms<min(.045,line_rms*.92)):
            choice='arc';model_arc=(cc,radius);residual=err
    cutoff=max(.035,3*np.median(residual)+.012)
    inliers=residual<=cutoff;fraction=float(np.mean(inliers))
    rms=float(np.sqrt(np.mean(residual[inliers]**2))) if np.any(inliers) else float('inf')
    info.update(fit_model=choice,fit_rmse_m=rms,inlier_fraction=fraction)
    if fraction<min_inlier_fraction or rms>fit_max_rmse_m:
        info['reason']='yellow ground curve fit unreliable';return None,None,info
    selected=points[inliers]
    if len(selected)<10 or np.linalg.norm(selected[-1]-selected[0])<.14:
        info['reason']='insufficient yellow support after outlier rejection';return None,None,info
    if choice=='arc':
        cc,radius=model_arc
        ang=np.unwrap(np.arctan2(selected[:,1]-cc[1],selected[:,0]-cc[0]))
        direction=1. if ang[-1]>ang[0] else -1.
        ang=np.maximum.accumulate(ang) if direction>0 else np.minimum.accumulate(ang)
        rvec=np.column_stack((np.cos(ang),np.sin(ang)))
        fitted=cc+radius*rvec
        tang=np.column_stack((-direction*rvec[:,1],direction*rvec[:,0]))
    else:
        t=(selected-median)@tangent;t=np.maximum.accumulate(t)
        fitted=median+t[:,None]*tangent
        tang=np.broadcast_to(tangent,fitted.shape)
    centre=fitted+.5*lane_width_m*np.column_stack((tang[:,1],-tang[:,0]))
    if not np.isfinite(centre).all():
        info['reason']='nonfinite ground reference';return None,None,info
    seglen=np.linalg.norm(np.diff(centre,axis=0),axis=1)
    if len(centre)<8 or np.sum(seglen)<.20:
        info['reason']='reconstructed centre arc shorter than 0.20 m';return None,None,info
    # Debug overlay follows true pinhole inverse projection for these fitted XY.
    xg,yg=centre[:,0],centre[:,1]
    sp,cp=math.sin(camera_pitch_rad),math.cos(camera_pitch_rad)
    denom=sp*xg+cp*camera_height_m
    numer=cp*xg-sp*camera_height_m
    pix_v=cy+fy*numer/np.maximum(denom,1.e-9)  # ray inverse via horizon geometry
    # Exact inverse based on forward / height:
    # (x/h)=(cp-down*sp)/(sp+down*cp) -> down=(cp-x*sp/h)/(sp+x*cp/h)
    down=(cp-xg*sp/camera_height_m)/(sp+xg*cp/camera_height_m)
    pix_v=cy+fy*down
    pix_u=cx-fx*yg*(sp+down*cp)/camera_height_m
    debug=[(float(u),float(v),0.0) for u,v in zip(pix_u,pix_v)
           if 0<=u<max(2*cx,820) and 0<=v<1000]
    visible_count=int(np.count_nonzero((pix_u>=0)&(pix_u<820)&(pix_v>=0)&(pix_v<410)))
    info.update(valid=True,reason='fresh calibrated ray-plane ground yellow / curve-normal offset',
                raw_span_m=float(np.sum(seglen)),pixels_used=projection['pixels_used'],
                debug_visible_points=visible_count,centre_off_camera=(visible_count==0))
    return centre,debug,info


def sanitize_recovery_ground_arclength(points, *, spacing_m=0.04,
                                      min_span_m=0.20,max_curvature_m_inv=2.5,
                                      max_observation_gap_m=0.25):
    """Validate metric path with ARC LENGTH; no monotonic x assumption.

    Preserve the near-field prefix; never bridge gaps or extrapolate outside
    the observed interval. Smooth high-frequency jitter before curvature test.
    """
    arr=np.asarray(points,float)
    info={'valid':False,'reason':'','raw_span_m':0.,'accepted_span_m':0.,
          'nearfield_only':False,'p95_curvature_m_inv':float('nan')}
    if arr.ndim!=2 or arr.shape[1]!=2 or len(arr)<8 or not np.isfinite(arr).all():
        info['reason']='invalid ground points';return None,info
    seg=np.linalg.norm(np.diff(arr,axis=0),axis=1)
    if len(seg)==0 or np.max(seg)>max_observation_gap_m:
        info['reason']='discontinuous observed yellow ground geometry';return None,info
    arclength=np.r_[0.,np.cumsum(seg)]
    good=np.r_[True,np.diff(arclength)>0.001]
    arr=arr[good];arclength=arclength[good]
    total=float(arclength[-1])
    info['raw_span_m']=total
    if total<min_span_m or len(arr)<8:
        info['reason']='insufficient ground arc length';return None,info
    # As before, far outlier sections can be shortened, not used to invent a path.
    for ratio in [1.,.90,.80,.70,.60,.50,.40]:
        finish=total*ratio
        if finish<min_span_m:break
        grid=np.linspace(0.,finish,max(8,int(finish/max(spacing_m,.025))+1))
        x=np.interp(grid,arclength,arr[:,0]);y=np.interp(grid,arclength,arr[:,1])
        # Light end-preserving smoothing without major shape change.
        if len(grid)>8:
            filt=np.array([1.,3.,5.,3.,1.])/13.
            xx=np.pad(x,(2,2),mode='edge');yy=np.pad(y,(2,2),mode='edge')
            x=np.convolve(xx,filt,'valid');y=np.convolve(yy,filt,'valid')
        path=np.column_stack((x,y))
        d=np.diff(path,axis=0);s=np.linalg.norm(d,axis=1)
        if len(s)<5 or np.min(s)<.004:continue
        headings=np.unwrap(np.arctan2(d[:,1],d[:,0]))
        cur=np.abs(np.diff(headings)/np.maximum(.5*(s[1:]+s[:-1]),1e-6))
        p95=float(np.percentile(cur,95))
        if np.isfinite(p95) and p95<=max_curvature_m_inv:
            info.update(valid=True,accepted_span_m=float(np.sum(s)),
                        p95_curvature_m_inv=p95,nearfield_only=(ratio<1.))
            return path,info
    info['reason']='recovery ground curvature exceeds configured safety bound'
    return None,info


def state_from_recovery_ground(path, heading_fit_m=0.35):
    a=np.asarray(path,float)
    if a.ndim!=2 or len(a)<8 or not np.isfinite(a).all():return None
    d=np.diff(a,axis=0);seg=np.linalg.norm(d,axis=1)
    s=np.r_[0.,np.cumsum(seg)];window=(s<=max(.15,heading_fit_m))
    if np.sum(window)<4:window[:min(8,len(s))]=True
    near=a[window]
    t=near[-1]-near[0]
    if np.linalg.norm(t)<.02:return None
    heading=math.atan2(t[1],t[0])
    hs=np.unwrap(np.arctan2(d[:,1],d[:,0]))
    curvature=float(np.median(np.diff(hs)/np.maximum(.5*(seg[1:]+seg[:-1]),1e-6)))
    return float(a[0,1]),-heading,curvature


def state_from_recovery_metric(path, heading_fit_m=0.35):
    """For RECOVERY debug only; original V23 _metric_state is never changed."""
    a=np.asarray(path,float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<5 or not np.isfinite(a).all():
        return None
    x,y=a[:,0],a[:,1]
    if x[-1]-x[0]<.20:return None
    near=x<=x[0]+max(.15,float(heading_fit_m))
    if np.count_nonzero(near)<4:near=np.arange(len(x))<min(len(x),8)
    slope=float(np.polyfit(x[near],y[near],1)[0])
    curve=np.polyfit(x,y,2)
    k=float(2*curve[0] / (1+slope*slope)**1.5)
    return float(y[0]),-math.atan(slope),k

def sanitize_recovery_nearfield(points, *, spacing_m=0.05, fit_radius_m=0.13,
                                min_span_m=0.20, max_curvature_m_inv=2.5,
                                max_observation_gap_m=0.25, max_trials=10):
    """Accept observed yellow recovery only where its NEAR segment is credible.

    In partial visibility, the far yellow trace may hook around the horizon or
    the cached perspective offset may be unreliable far from the camera.
    Validate the full metric path first. If that fails due to excessive
    curvature or an observation gap, attempt progressively SHORTER contiguous
    near-to-far prefixes. Every prefix goes through the unchanged ORIGINAL
    sanitizer using the same curvature and gap limits. Never extrapolate,
    relax curvature limits, choose disconnected samples or skip a bad near
    segment. This is recovery-only; FULL path processing is unaffected.

    Returns (cleaned_path, diagnostic) with trimmed/reached fields.
    """
    arr=np.asarray(points,dtype=float)
    details={'valid':False, 'reason':'invalid recovery geometry',
             'raw_span_m':0.0, 'accepted_span_m':0.0,
             'trimmed_far_end_m':0.0, 'nearfield_only':False,
             'trials':0}
    if arr.ndim!=2 or arr.shape[1]!=2 or len(arr)<6 or not np.isfinite(arr).all():
        return None,details
    # The _metric_state method already builds monotonically increasing forward
    # samples; checking them here prevents a cut from hiding any near-field gap.
    if np.any(np.diff(arr[:,0])<=0):
        details['reason']='nonmonotonic recovery forward samples'
        return None,details
    details['raw_span_m']=float(arr[-1,0]-arr[0,0])
    kwargs=dict(spacing_m=spacing_m, fit_radius_m=fit_radius_m,
                min_span_m=min_span_m, max_curvature_m_inv=max_curvature_m_inv,
                max_observation_gap_m=max_observation_gap_m)
    cleaned, info=sanitize_metric_lane_path(arr,**kwargs)
    details['trials']=1
    details.update({k:info.get(k,details.get(k)) for k in
                    ('reason','p95_curvature_m_inv','span_m')})
    if cleaned is not None:
        details.update(valid=True, accepted_span_m=float(cleaned[-1,0]-cleaned[0,0]))
        return cleaned,details
    # Do not attempt shortening for a missing near-field path; shortening does
    # not manufacture new observations and must never bypass any hard limit.
    if info.get('reason') not in ('observed path curvature exceeds configured safety bound',
                                  'gap between observed lane samples is too large'):
        return None,details
    if details['raw_span_m']<min_span_m+0.055:
        return None,details
    # Work from longest to shortest; keep all samples from the NEAREST point.
    # Limit the number of expensive local fits on resource-constrained QCar PCs.
    end=float(arr[-1,0]); start=float(arr[0,0])
    raw_span=end-start
    candidates=np.linspace(end-0.055,start+min_span_m+0.015,max(2,int(max_trials)-1))
    for cutoff in candidates:
        if cutoff <= start+min_span_m:continue
        subset=arr[arr[:,0]<=cutoff]
        if len(subset)<6 or subset[-1,0]-subset[0,0]<min_span_m:continue
        details['trials']+=1
        maybe, subinfo=sanitize_metric_lane_path(subset,**kwargs)
        if maybe is None: continue
        details.update(valid=True, nearfield_only=True,
                       reason='accepted only observed near-field prefix',
                       accepted_span_m=float(maybe[-1,0]-maybe[0,0]),
                       trimmed_far_end_m=float(raw_span-(subset[-1,0]-subset[0,0])),
                       p95_curvature_m_inv=float(subinfo['p95_curvature_m_inv']),
                       span_m=float(subinfo['span_m']))
        return maybe,details
    details['reason']='near-field also fails original curvature/gap limits: '+str(info.get('reason'))
    return None,details


def compare_recovery_to_last_full(candidate_world, full_world, *,
                                  max_distance_m=0.12, max_heading_error_rad=0.45,
                                  min_match_fraction=0.65, min_match_span_m=0.10):
    """Reject a guessed recovery corridor inconsistent with last measured FULL path.

    Candidate must lie near the measured world polyline; no path extension is
    authorized outside it. A recovered geometry need not match every point,
    but the matched section must have substantial measured forward reach.
    """
    a,b = np.asarray(candidate_world,float),np.asarray(full_world,float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (2,) or b.shape[1:] != (2,):
        return False,'invalid path shape'
    if len(a)<5 or len(b)<5 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False,'invalid/nonfinite path'
    seg = np.diff(b,axis=0)
    len2 = np.einsum('ij,ij->i',seg,seg)
    if not np.any(len2>1e-6):
        return False,'degenerate previous path'
    matched=[]
    differences=[]
    for i,pt in enumerate(a):
        fraction=np.clip(np.einsum('ij,ij->i',pt[None,:]-b[:-1],seg)/np.maximum(len2,1e-9),0,1)
        projection=b[:-1]+fraction[:,None]*seg
        residual=np.linalg.norm(projection-pt,axis=1)
        j=int(np.argmin(residual))
        if residual[j]>max_distance_m:
            continue
        matched.append(i)
        if 0<i<len(a)-1:
            tangent=a[i+1]-a[i-1]
            if np.linalg.norm(tangent)>1e-5 and len2[j]>1e-6:
                angle=math.atan2(tangent[0]*seg[j,1]-tangent[1]*seg[j,0],
                                 float(np.dot(tangent,seg[j])))
                differences.append(abs(angle))
    if len(matched)<max(5,math.ceil(min_match_fraction*len(a))):
        return False,'recovered path has insufficient correspondence to FULL'
    if len(matched)<2 or sum(float(np.linalg.norm(a[j]-a[i])) for i,j in zip(matched[:-1],matched[1:]))<min_match_span_m:
        return False,'recovered path has insufficient overlap'
    if not differences or np.percentile(differences,90)>max_heading_error_rad:
        return False,'recovered heading differs from FULL'
    return True,'matched previous FULL path'


def median_filter_1d(values, radius=3):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    out = values.copy()
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


class LanePathDelayBuffer:
    """Bounded FIFO for artificially delayed *validated* camera paths.

    Scheduling uses steady/wall time rather than a ROS simulation clock, matching
    perception transport latency in physical seconds. No path is edited or
    synthesized. This class does not perform control or perception.
    """

    def __init__(self, delay_ms, max_pending=64):
        delay_ms = float(delay_ms)
        if not math.isfinite(delay_ms) or delay_ms < 0.0:
            raise ValueError('perception_delay_ms must be finite and >= 0')
        if int(max_pending) < 1:
            raise ValueError('max_pending must be positive')
        self.delay_sec = delay_ms / 1000.0
        self.max_pending = int(max_pending)
        self._items = deque()
        self.dropped = 0

    def push(self, path, now):
        """Queue a path, preserving its ROS stamp and coordinates verbatim."""
        now = float(now)
        if not math.isfinite(now):
            raise ValueError('nonfinite monotonic time')
        if len(self._items) >= self.max_pending:
            self._items.popleft()  # Never release stale data early to catch up.
            self.dropped += 1
        self._items.append((now + self.delay_sec, now, path))

    def next_release_time(self):
        return self._items[0][0] if self._items else None

    def pop_ready(self, now):
        now = float(now)
        ready = []
        while self._items and self._items[0][0] <= now:
            due, enqueue_time, path = self._items.popleft()
            ready.append((path, 1000.0 * (now - enqueue_time)))
        return ready

    def clear(self):
        self._items.clear()


class LaneDetector(Node):
    """camera lane detector and ROS path publisher.

    ROS parameters
    --------------
    Camera and road mask:
        ``image_topic`` selects camera frames; ``camera_info_topic`` supplies
        camera intrinsics; ``speed_topic`` selects the optional speed input.
        ``roi_top_frac`` is the fraction of the image cropped from the top.
        ``road_threshold_offset`` adjusts the adaptive dark-road threshold,
        bounded by ``road_threshold_min`` and ``road_threshold_max``.
        ``road_seed_half_width_frac`` and ``road_seed_top_frac`` set the
        bottom-centre seed region for the connected road component.
        ``road_close_width_px`` and ``road_close_height_px`` set the
        morphological closing kernel; ``road_min_component_area_frac`` is
        the minimum accepted road-component area as a fraction of the ROI.

    Yellow left-boundary detection:
        ``yellow_hue_min`` / ``yellow_hue_max``,
        ``yellow_saturation_min`` and ``yellow_value_min`` set HSV thresholds.
        ``yellow_min_run_px`` / ``yellow_max_run_px`` bound a scanline run.
        ``yellow_start_max_x_frac`` limits the initial search position.
        ``yellow_component_min_area`` and ``yellow_component_min_span_px``
        reject small components; ``yellow_road_dilate_px`` expands the road
        mask when checking yellow-road overlap; ``yellow_max_border_touch_frac``
        rejects components touching too much of the image border.
        ``yellow_max_row_jump_px`` limits jumps while following the component.
        ``yellow_far_seed_top_fraction`` and
        ``yellow_far_seed_min_x_frac`` constrain the far-end tracking seed.
        ``yellow_component_min_aspect_span`` rejects compact components;
        ``yellow_component_bottom_bonus`` biases component selection toward
        evidence near the vehicle.
        ``yellow_path_max_gap_rows`` and ``yellow_path_max_jump_px`` limit
        missing rows and lateral jumps in the smooth path tracker.
        ``yellow_path_jump_penalty`` and ``yellow_path_gap_penalty`` penalize
        jumps and gaps in candidate paths; ``yellow_path_road_bonus`` rewards
        road-supported candidates.
        ``yellow_path_temporal_bonus`` weights agreement with the previous
        frame, and ``yellow_path_temporal_scale_px`` sets its pixel distance
        scale. ``yellow_path_min_score`` is the minimum accepted path score;
        ``yellow_hold_last_sec`` sets how long a recent yellow trace may be held.

    Right-boundary detection and corridor checks:
        ``right_search_min_x_frac`` / ``right_search_max_x_frac`` bound the
        horizontal search region; ``right_border_exclusion_px`` excludes the
        image border. ``right_gradient_min``, ``right_contrast_min`` and
        ``right_bright_min`` set gradient, local contrast and brightness
        requirements for the road edge.
        ``right_of_yellow_margin_px`` requires separation from the yellow
        boundary; ``right_max_row_jump_px`` limits row-to-row tracking jumps.
        ``right_min_road_pixels_left`` and ``right_min_support_rows`` set
        minimum road support and supporting scan rows;
        ``right_ignore_top_frac_of_roi`` skips the upper fraction of the ROI.
        ``right_white_fallback`` enables the Gazebo white-stripe fallback.
        ``right_white_value_min`` / ``right_white_saturation_max`` define its
        bright, low-saturation mask; ``right_white_min_area_px`` and
        ``right_white_min_span_frac`` require sufficient component size and
        vertical span; ``right_white_max_band_width_frac`` caps stripe width;
        ``right_white_min_asphalt_support`` requires adjacent asphalt support.
        ``require_both_boundaries`` gates a valid corridor on observing both
        sides. ``right_min_overlap_roi_frac`` sets the minimum shared vertical
        coverage; ``right_max_near_gap_roi_frac`` caps the near-field gap.

    Scanline geometry and metric state:
        ``scan_row_step`` is the vertical pixel interval between scanlines.
        ``max_missing_rows`` limits consecutive unsupported rows;
        ``minimum_boundary_points`` and ``minimum_support_span_px`` set
        minimum point count and pixel span for a boundary.
        ``lane_width_m`` is the physical lane width used in metric conversion
        and one-sided fallback. ``default_lane_width_bottom_frac`` and
        ``default_lane_width_top_frac`` define its image-width profile;
        ``lane_width_alpha`` controls smoothing of the width estimate.
        ``visible_forward_range_m`` sets the metric extent;
        ``heading_fit_distance_m`` and ``curvature_fit_distance_m`` set the
        local fitting distances for heading and curvature.
        ``straight_rmse_m`` and ``straight_heading_change_rad`` are the fit
        error and heading-change limits for classifying a path as straight.
        ``cte_filter_alpha``, ``heading_filter_alpha`` and
        ``curvature_filter_alpha`` smooth their respective reported signals.
        ``max_abs_cte_m``, ``max_abs_heading_rad`` and ``max_abs_curvature``
        bound valid cross-track error, heading error and curvature.

    ROS path publishing and camera geometry:
        ``odom_topic`` selects odometry; ``lane_path_topic`` and
        ``lane_path_quality_topic`` select the path and quality outputs.
        ``perception_delay_ms`` delays published paths for latency testing.
        ``camera_x_from_base_m``, ``camera_y_from_base_m`` and
        ``camera_yaw_from_base_rad`` locate and orient the camera relative to
        the base link. ``odom_freshness_sec`` is the maximum odometry age;
        ``min_path_points`` and ``min_path_forward_span_m`` set minimum path
        point count and observed forward extent.
        ``require_camera_info_for_path`` gates metric path publication on
        calibrated camera intrinsics. ``min_lane_status_for_path`` sets the
        required lane status (for example, 3 for FULL); ``publish_lane_path``
        enables or disables the path output.

    Single-boundary recovery:
        ``ground_camera_height_m`` and ``ground_camera_pitch_rad`` configure
        the ground-plane projection; ``ground_max_recovery_range_m`` caps its
        range. ``enable_single_boundary_recovery`` enables this recovery path.
        ``recovery_max_age_sec`` limits source-frame age;
        ``recovery_curve_fit_max_rmse_m`` and
        ``recovery_curve_min_inlier_fraction`` set curve-fit quality limits.
        ``recovery_visual_max_age_sec`` limits yellow-only visual-path age;
        ``recovery_max_world_distance_m`` limits its distance from the last
        calibrated path.
        ``recovery_drive_max_age_sec``, ``recovery_drive_max_travel_m`` and
        ``recovery_drive_min_span_m`` constrain drive-eligible path age,
        travel and observed span; ``recovery_drive_min_consecutive`` sets the
        number of consecutive qualifying frames.
        ``recovery_ground_geometry_verified`` is the explicit opt-in asserting
        ground geometry has been physically verified.
        ``recovery_drive_max_abs_cte_m`` and
        ``recovery_drive_max_abs_heading_rad`` bound drive-eligible error;
        ``recovery_max_heading_difference_rad`` limits heading change from the
        last full observation.

    Published path smoothing and debug transport:
        ``lane_path_spacing_m`` sets resampling interval;
        ``lane_path_fit_radius_m`` sets the local smoothing radius;
        ``lane_path_min_span_m`` and ``lane_path_max_curvature_m_inv`` reject
        paths that are too short or sharply curved;
        ``lane_path_max_observation_gap_m`` rejects interpolation across a
        large unobserved gap.
        ``debug`` enables debug outputs. ``publish_raw_debug`` and
        ``publish_compressed_debug`` independently enable raw and JPEG images.
        ``debug_jpeg_quality`` sets JPEG quality;
        ``compressed_debug_width`` sets output width (zero keeps source width);
        ``raw_debug_publish_divisor`` publishes one raw image every N frames.
    """

    def __init__(self):
        super().__init__('lane_detector')

        # ------------------------------------------------------------
        # Camera / ROI
        # ------------------------------------------------------------
        self.declare_parameter('image_topic', '/front_camera/image_raw')
        self.declare_parameter('camera_info_topic', '/front_camera/camera_info')
        self.declare_parameter('speed_topic', '/state/velocity')
        self.declare_parameter('roi_top_frac', 0.48)

        # ------------------------------------------------------------
        # V18 road-surface component.
        #
        # The road is selected as the dark connected component attached to a
        # seed near the vehicle/bottom-centre of the camera image. This is the
        # key difference from V17: walls and furniture may contain strong
        # gradients, but they are not connected to the asphalt under QCar.
        # ------------------------------------------------------------
        self.declare_parameter('road_threshold_offset', 8.0)
        self.declare_parameter('road_threshold_min', 65.0)
        self.declare_parameter('road_threshold_max', 150.0)
        self.declare_parameter('road_seed_half_width_frac', 0.16)
        self.declare_parameter('road_seed_top_frac', 0.72)
        self.declare_parameter('road_close_width_px', 21)
        self.declare_parameter('road_close_height_px', 7)
        self.declare_parameter('road_min_component_area_frac', 0.08)

        # ------------------------------------------------------------
        # Yellow LEFT boundary
        # ------------------------------------------------------------
        self.declare_parameter('yellow_hue_min', 10)
        self.declare_parameter('yellow_hue_max', 45)
        self.declare_parameter('yellow_saturation_min', 30)
        self.declare_parameter('yellow_value_min', 50)
        self.declare_parameter('yellow_min_run_px', 2)
        self.declare_parameter('yellow_max_run_px', 65)
        self.declare_parameter('yellow_start_max_x_frac', 0.86)
        self.declare_parameter('yellow_component_min_area', 25)
        self.declare_parameter('yellow_component_min_span_px', 24.0)
        self.declare_parameter('yellow_road_dilate_px', 18)
        self.declare_parameter('yellow_max_border_touch_frac', 0.18)

        # V19 yellow tracking: start from the FAR/top end of the visible
        # yellow component, then follow it toward the vehicle. This prevents a
        # small bottom-left reflection/branch from stealing the tracker.
        self.declare_parameter('yellow_max_row_jump_px', 32.0)
        self.declare_parameter('yellow_far_seed_top_fraction', 0.45)
        self.declare_parameter('yellow_far_seed_min_x_frac', 0.20)

        # V20 dominant-component selection.
        self.declare_parameter('yellow_component_min_aspect_span', 0.8)
        self.declare_parameter('yellow_component_bottom_bonus', 120.0)

        # V21 yellow smooth-path tracker.
        self.declare_parameter('yellow_path_max_gap_rows', 8)
        self.declare_parameter('yellow_path_max_jump_px', 42.0)
        self.declare_parameter('yellow_path_jump_penalty', 0.055)
        self.declare_parameter('yellow_path_gap_penalty', 0.65)
        self.declare_parameter('yellow_path_road_bonus', 2.0)
        self.declare_parameter('yellow_path_temporal_bonus', 2.5)
        self.declare_parameter('yellow_path_temporal_scale_px', 45.0)
        self.declare_parameter('yellow_path_min_score', 18.0)
        self.declare_parameter('yellow_hold_last_sec', 0.18)

        # ------------------------------------------------------------
        # RIGHT physical road/white boundary.
        #
        # We deliberately track the dark->bright edge instead of
        # thresholding a thin white stripe.  In the physical camera the
        # white line often merges visually into the pale curb.
        # ------------------------------------------------------------
        self.declare_parameter('right_search_min_x_frac', 0.42)
        self.declare_parameter('right_search_max_x_frac', 0.99)
        self.declare_parameter('right_border_exclusion_px', 8)

        self.declare_parameter('right_gradient_min', 10.0)
        self.declare_parameter('right_contrast_min', 10.0)
        self.declare_parameter('right_bright_min', 80.0)

        # Must be to the right of yellow when yellow exists at that row.
        self.declare_parameter('right_of_yellow_margin_px', 18)
        self.declare_parameter('right_max_row_jump_px', 38.0)
        self.declare_parameter('right_min_road_pixels_left', 12)
        self.declare_parameter('right_min_support_rows', 10)
        self.declare_parameter('right_ignore_top_frac_of_roi', 0.08)

        # Gazebo paints a discrete white stripe against black asphalt. The
        # original V23 road-edge tracker is retained for hardware; only if it
        # fails do we seek the INSIDE (left) edge of a supported white stripe.
        # Safe default: require a large, continuous, low-saturation, bright
        # component to the RIGHT of the observed yellow boundary.
        self.declare_parameter('right_white_fallback', True)
        self.declare_parameter('right_white_value_min', 165)
        self.declare_parameter('right_white_saturation_max', 75)
        self.declare_parameter('right_white_min_area_px', 100)
        self.declare_parameter('right_white_min_span_frac', 0.20)
        self.declare_parameter('right_white_max_band_width_frac', 0.18)
        self.declare_parameter('right_white_min_asphalt_support', 0.65)
        # Only a real two-sided corridor is a valid driving reference.
        # Leave disabled for legacy scalar-only one-edge experiments, not driving.
        self.declare_parameter('require_both_boundaries', True)
        self.declare_parameter('right_min_overlap_roi_frac', 0.18)
        self.declare_parameter('right_max_near_gap_roi_frac', 0.22)

        # ------------------------------------------------------------
        # Shared tracking geometry
        # ------------------------------------------------------------
        self.declare_parameter('scan_row_step', 2)
        self.declare_parameter('max_missing_rows', 12)
        self.declare_parameter('minimum_boundary_points', 10)
        self.declare_parameter('minimum_support_span_px', 28.0)

        # Physical lane width used only for metric conversion / one-side fallback.
        self.declare_parameter('lane_width_m', 0.25)
        self.declare_parameter('default_lane_width_bottom_frac', 0.72)
        self.declare_parameter('default_lane_width_top_frac', 0.32)
        self.declare_parameter('lane_width_alpha', 0.12)

        # ------------------------------------------------------------
        # Metric state
        # ------------------------------------------------------------
        self.declare_parameter('visible_forward_range_m', 1.00)
        self.declare_parameter('heading_fit_distance_m', 0.35)
        self.declare_parameter('curvature_fit_distance_m', 0.80)

        self.declare_parameter('straight_rmse_m', 0.025)
        self.declare_parameter('straight_heading_change_rad', 0.12)

        self.declare_parameter('cte_filter_alpha', 0.50)
        self.declare_parameter('heading_filter_alpha', 0.40)
        self.declare_parameter('curvature_filter_alpha', 0.30)

        self.declare_parameter('max_abs_cte_m', 0.70)
        self.declare_parameter('max_abs_heading_rad', 3.24)
        self.declare_parameter('max_abs_curvature', 3.0)

        # Rcognita ROS lane-mode interface. Leave the V23 tracker unchanged:
        # publish its directly measured centerline as a path in the odom frame.
        self.declare_parameter('odom_topic', '/odom/bot')
        self.declare_parameter('lane_path_topic', '/lane_detection/reference_path')
        self.declare_parameter('lane_path_quality_topic', '/lane_detection/path_quality')
        # Optional disturbance from lane_detection_vf: delay only published paths.
        self.declare_parameter('perception_delay_ms', 0.0)
        self.declare_parameter('camera_x_from_base_m', 0.19236)
        self.declare_parameter('camera_y_from_base_m', -0.000475)
        self.declare_parameter('camera_yaw_from_base_rad', 0.0)
        self.declare_parameter('odom_freshness_sec', 0.35)
        self.declare_parameter('min_path_points', 8)
        self.declare_parameter('min_path_forward_span_m', 0.10)
        # A camera-ground metric path should not rely on the uncalibrated
        # row-to-distance fallback. Diagnostic scalar topics still work.
        self.declare_parameter('require_camera_info_for_path', True)
        # Safety default: publish a drivable path only after BOTH sides are detected.
        # Set to 2 explicitly only after one-side fallback has been validated.
        self.declare_parameter('min_lane_status_for_path', 3)
        self.declare_parameter('publish_lane_path', True)
        # Isolated yellow-only recovery, not part of normal V23 tracking.
        # Physical recovery geometry: preliminary 2026-10-27 marker-based estimate.
        # Camera lens 0.085 m high; pitch approx -0.023 rad from 0.2/0.4/0.6 m markers.
        # Reconfirm physical geometry; positive pitch points down.
        self.declare_parameter('ground_camera_height_m', 0.085)
        self.declare_parameter('ground_camera_pitch_rad', -0.023)
        self.declare_parameter('ground_max_recovery_range_m', 1.40)
        self.declare_parameter('enable_single_boundary_recovery', True)
        self.declare_parameter('recovery_max_age_sec', 0.30)
        self.declare_parameter('recovery_curve_fit_max_rmse_m', 0.055)
        self.declare_parameter('recovery_curve_min_inlier_fraction', 0.80)
        # Sensor-only extended recovery: a NEW path is built from yellow in
        # EVERY PARTIAL frame. This never extends the relay's motor gate.
        self.declare_parameter('recovery_visual_max_age_sec', 15.0)
        self.declare_parameter('recovery_max_world_distance_m', 0.12)
        # FULL-calibrated metric yellow recovery. Physical driving additionally
        # requires an independent opt-in at the relay and stricter checks.
        self.declare_parameter('recovery_drive_max_age_sec', 1.50)
        self.declare_parameter('recovery_drive_max_travel_m', 0.45)
        self.declare_parameter('recovery_drive_min_span_m', 0.30)
        self.declare_parameter('recovery_drive_min_consecutive', 4)
        # Physical-drive permission is NEVER inferred from a provisional pitch.
        # Set TRUE only after physically validating ground projection/steering.
        # Keeping this FALSE leaves yellow-only ROS Paths available for observation.
        self.declare_parameter('recovery_ground_geometry_verified', False)
        self.declare_parameter('recovery_drive_max_abs_cte_m', 0.095)
        self.declare_parameter('recovery_drive_max_abs_heading_rad', 0.35)
        self.declare_parameter('recovery_max_heading_difference_rad', 0.45)
        # MPC requires smooth metric geometry, but the original V23 pixel
        # tracking and raw scalar/debug topics are intentionally unchanged.
        self.declare_parameter('lane_path_spacing_m', 0.025)
        self.declare_parameter('lane_path_fit_radius_m', 0.13)
        self.declare_parameter('lane_path_min_span_m', 0.10)
        self.declare_parameter('lane_path_max_curvature_m_inv', 10.0)
        self.declare_parameter('lane_path_max_observation_gap_m', 0.40)

        self.declare_parameter('debug', True)

        # V23 debug transport.
        # Keep the original raw debug image, and additionally publish a JPEG
        # stream for low-bandwidth / low-latency visualization in rqt.
        self.declare_parameter('publish_raw_debug', True)
        self.declare_parameter('publish_compressed_debug', True)
        self.declare_parameter('debug_jpeg_quality', 55)
        # 0 means keep original width. 640 is a good hardware/PC compromise.
        self.declare_parameter('compressed_debug_width', 640)
        # Raw remains available, but publishing it every Nth processed frame
        # prevents the large uncompressed stream from competing with the
        # compressed live view. Set to 1 if full-rate raw is required.
        self.declare_parameter('raw_debug_publish_divisor', 3)

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self._bridge = CvBridge()
        self._camera_fx = None
        self._camera_fy = None
        self._camera_cx = None
        self._camera_cy = None
        self._speed = 0.0
        # Cache the metric samples calculated for the CURRENT frame only.
        self._current_metric_path = None  # Nx2 [forward, left] in camera ground frame
        self._right_source = 'none'
        self._right_span_m = 0.0
        self._right_rejection_details = ''

        self._odom_pose = None            # (base_x, base_y, yaw, odom frame)
        self._last_odom_wall_time = None  # wall monotonic; works if /clock is absent
        self._odom_lock = threading.Lock()
        self._odom_callback_group = MutuallyExclusiveCallbackGroup()
        self._odom_received_count = 0
        self._last_path_skip_log_time = float('-inf')

        self._previous_left_x = None
        self._previous_right_x = None

        # V21 temporal prior for the real yellow lane path.
        self._last_left_points = []
        self._last_left_points_time = None

        self._learned_width_near = None
        self._learned_width_far = None

        self._filtered_cte = None
        self._filtered_heading = None
        self._filtered_curvature = None

        self._last_valid_time = None
        # History comes only from an ACCEPTED, measured two-boundary path.
        self._recovery_full_offset_profile = None  # [image_row, half-width_px]
        self._recovery_ground_scale_profile = None  # [image_row, full_width_px] FULL-only
        self._recovery_consecutive = 0
        self._recovery_travel_m = 0.0
        self._recovery_prev_odom_xy = None
        # Monotonic receipt timestamp paired with the last recovery XY.
        # A FULL -> PARTIAL gap must be judged using elapsed time, not a
        # fixed 20 cm limit that treats normal 0.5 m/s travel as a jump.
        self._recovery_prev_odom_wall_time = None
        self._recovery_full_world_path = None       # world path paired with last VALID calibration
        self._latest_accepted_full_world_path = None  # most recent FULL path (may lack profile)
        self._recovery_calibration_reject_reason = 'no FULL geometry calibrated yet'
        self._recovery_last_calibration_warn = float('-inf')
        self._recovery_full_time = None
        self._recovery_full_frame = None
        self._recovery_full_image_width = None
        self._recovery_prev_world_path = None
        self._recovery_prev_world_time = None
        self._debug_frame_counter = 0

        # ------------------------------------------------------------
        # ROS
        # ------------------------------------------------------------
        cam_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._image_sub = self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self._image_cb,
            cam_qos,
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self._camera_info_cb,
            10,
        )
        self._speed_sub = self.create_subscription(
            Float64,
            str(self.get_parameter('speed_topic').value),
            self._speed_cb,
            10,
        )
        # A separate executor group allows odometry callbacks to run while
        # the CPU-heavy image-processing callback is still executing. Without
        # this, the 0.35 s odometry freshness guard can reject live odometry.
        self._odom_sub = self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._odom_cb, 10, callback_group=self._odom_callback_group,
        )
        # Camera-derived paths are a live stream, not a latched global route.
        self._lane_path_pub = self.create_publisher(
            ROSPath, str(self.get_parameter('lane_path_topic').value), 10,
        )
        self._lane_path_quality_pub = self.create_publisher(
            String, str(self.get_parameter('lane_path_quality_topic').value), 10,
        )
        # Optional release worker runs independently of expensive CV image
        # callbacks: an executor timer could add a whole frame of extra delay.
        # It does NOT run a vehicle control loop. It only releases validated
        # nav_msgs/Path objects unchanged on the existing lane_path_topic.
        self._perception_delay_ms = float(self.get_parameter('perception_delay_ms').value)
        if not math.isfinite(self._perception_delay_ms) or self._perception_delay_ms < 0.0:
            raise ValueError('perception_delay_ms must be a nonnegative finite number')
        self._path_delay_buffer = None
        self._path_delay_cv = None
        self._path_delay_thread = None
        self._path_delay_shutdown = False
        self._actual_delay_pub = None
        if self._perception_delay_ms > 0.0:
            self._path_delay_buffer = LanePathDelayBuffer(self._perception_delay_ms)
            self._path_delay_cv = threading.Condition()
            self._actual_delay_pub = self.create_publisher(
                Float64, '/lane_detection/actual_perception_delay_ms', 10,
            )
            self._path_delay_thread = threading.Thread(
                target=self._release_delayed_paths,
                name='qcar_lane_path_delay',
                daemon=True,
            )
            self._path_delay_thread.start()
            self.get_logger().info(
                'Thesis perception disturbance ENABLED: +%.0f ms delay to %s; '
                'status/CTE/debug remain immediate; original Path stamps retained.' % (
                    self._perception_delay_ms,
                    str(self.get_parameter('lane_path_topic').value),
                )
            )

        # Separate, correlated proof; PARTIAL never impersonates FULL status.
        self._recovery_evidence_pub = self.create_publisher(
            String, '/lane_detection/recovery_evidence', 10)
        self._recovery_valid_pub = self.create_publisher(
            Float64, '/lane_detection/recovery_valid', 10)
        # Detailed reason codes for rejected PARTIAL recovery. Sensor-only debug.
        self._recovery_diagnostic_pub = self.create_publisher(
            String, '/lane_detection/recovery_diagnostics', 10)
        self._recovery_last_reason = 'no valid FULL path cached'


        self._cte_pub = self.create_publisher(Float64, '/control/cte', 10)
        self._heading_pub = self.create_publisher(
            Float64, '/control/heading_error', 10
        )
        self._curvature_pub = self.create_publisher(
            Float64, '/control/curvature', 10
        )
        self._status_pub = self.create_publisher(
            Float64, '/lane_detection/status', 10
        )

        # Low-latency visualization QoS: never queue old debug frames.
        debug_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._debug_pub = self.create_publisher(
            Image, '/lane_detection/debug_image', debug_qos
        )
        self._debug_compressed_pub = self.create_publisher(
            CompressedImage,
            '/lane_detection/debug_image/compressed',
            debug_qos,
        )
        self._yellow_mask_pub = self.create_publisher(
            Image, '/lane_detection/yellow_mask', 2
        )
        self._right_edge_pub = self.create_publisher(
            Image, '/lane_detection/right_edge_mask', 2
        )
        self._road_mask_pub = self.create_publisher(
            Image, '/lane_detection/road_mask', 2
        )
        self._corridor_pub = self.create_publisher(
            Image, '/lane_detection/corridor_mask', 2
        )

        self.get_logger().info(
            'QCar lane detector V23 + measured-height recovery started: direct scanline tracking; '
            'LEFT=global smooth yellow run path with temporal continuity, RIGHT=edge of bottom-connected asphalt; raw + JPEG-compressed debug streams enabled.'
        )

    # ==================================================================
    # ROS callbacks
    # ==================================================================
    def _camera_info_cb(self, msg):
        try:
            fx = float(msg.k[0])
            fy, cx, cy = float(msg.k[4]), float(msg.k[2]), float(msg.k[5])
            if (math.isfinite(fx) and fx > 1.0 and math.isfinite(fy) and fy > 1.0
                    and math.isfinite(cx) and math.isfinite(cy)):
                self._camera_fx=fx
                self._camera_fy=fy
                self._camera_cx=cx
                self._camera_cy=cy
        except Exception:
            pass

    def _speed_cb(self, msg):
        value = float(msg.data)
        if math.isfinite(value):
            self._speed = value

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        values = (float(p.x), float(p.y), float(yaw))
        if not all(math.isfinite(v) for v in values):
            return
        frame = str(msg.header.frame_id)
        if not frame:
            return
        # Assumes odometry pose is for the URDF base link. Verify this first.
        with self._odom_lock:
            self._odom_pose = (*values, frame)
            self._last_odom_wall_time = time.monotonic()
            self._odom_received_count += 1

    def _odom_snapshot(self):
        """Return an atomic snapshot of the last processed odometry message."""
        with self._odom_lock:
            return (self._odom_pose, self._last_odom_wall_time,
                    self._odom_received_count)

    def _image_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(
                'cv_bridge conversion failed: {}'.format(exc),
            )
            return

        self._process_frame(frame, msg.header.frame_id or 'front_camera')

    # ==================================================================
    # Masks
    # ==================================================================
    def _road_component(self, roi_bgr):
        """Return the dark asphalt component connected to bottom-centre QCar.

        Unlike a generic threshold mask, this rejects dark wall/furniture
        regions that are not physically connected to the drivable asphalt.
        """
        p = lambda name: self.get_parameter(name).value

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        otsu, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = clamp(
            float(otsu) + float(p('road_threshold_offset')),
            float(p('road_threshold_min')),
            float(p('road_threshold_max')),
        )

        road = np.zeros_like(gray, dtype=np.uint8)
        road[gray <= threshold] = 255

        kw = max(3, int(p('road_close_width_px')))
        kh = max(3, int(p('road_close_height_px')))
        if kw % 2 == 0:
            kw += 1
        if kh % 2 == 0:
            kh += 1

        # Bridge the painted yellow stripe / small reflections inside asphalt.
        road = cv2.morphologyEx(
            road,
            cv2.MORPH_CLOSE,
            np.ones((kh, kw), dtype=np.uint8),
            iterations=1,
        )
        road = cv2.morphologyEx(
            road,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            road, connectivity=8
        )
        if n <= 1:
            return None

        h, w = gray.shape[:2]
        seed_half = max(
            10,
            int(float(p('road_seed_half_width_frac')) * w),
        )
        seed_y0 = int(
            clamp(float(p('road_seed_top_frac')), 0.40, 0.95) * h
        )
        seed_x0 = max(0, w // 2 - seed_half)
        seed_x1 = min(w, w // 2 + seed_half)

        min_area = max(
            100,
            int(float(p('road_min_component_area_frac')) * h * w),
        )

        best_label = None
        best_score = None

        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component = labels == label
            seed_overlap = int(
                np.count_nonzero(
                    component[seed_y0:h, seed_x0:seed_x1]
                )
            )
            if seed_overlap <= 0:
                continue

            ys, xs = np.where(component)
            if xs.size == 0:
                continue

            lower_overlap = int(
                np.count_nonzero(ys >= int(0.70 * h))
            )
            y_span = float(np.ptp(ys)) if ys.size else 0.0

            score = (
                8.0 * float(seed_overlap)
                + float(area)
                + 2.0 * float(lower_overlap)
                + 2.0 * y_span
            )

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        selected = np.zeros_like(gray, dtype=np.uint8)
        selected[labels == best_label] = 255
        return selected

    def _select_yellow_component(self, yellow_mask, road_mask):
        """Select the dominant elongated yellow painted-lane component.

        V18/V19 made road overlap a hard requirement. On a curve the yellow
        paint is bright and can sit just outside the thresholded dark-asphalt
        component, so the genuine stripe was sometimes rejected while a small
        yellow-ish patch near the ROI survived.

        V20 uses the properties that are actually visible in the camera:
        the genuine yellow lane is a long, elongated connected component with
        substantial vertical span. Road proximity is only a bonus, not a gate.
        """
        if yellow_mask is None or yellow_mask.size == 0:
            return np.zeros_like(yellow_mask)

        p = lambda name: self.get_parameter(name).value
        h, w = yellow_mask.shape[:2]

        # A little more closing than V19 so a curved painted stripe with small
        # lighting gaps stays one connected component.
        work = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=2,
        )
        work = cv2.morphologyEx(
            work,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            work, connectivity=8
        )
        if n <= 1:
            return np.zeros_like(yellow_mask)

        road_near = None
        if road_mask is not None:
            dilate_px = max(1, int(p('yellow_road_dilate_px')))
            k = 2 * dilate_px + 1
            road_near = cv2.dilate(
                road_mask,
                np.ones((k, k), dtype=np.uint8),
                iterations=1,
            ) > 0

        min_area = max(5, int(p('yellow_component_min_area')))
        min_span = max(8.0, float(p('yellow_component_min_span_px')))
        max_border_frac = clamp(
            float(p('yellow_max_border_touch_frac')), 0.0, 1.0
        )
        bottom_bonus_scale = max(
            0.0, float(p('yellow_component_bottom_bonus'))
        )

        best_label = None
        best_score = None

        for label in range(1, n):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < min_area:
                continue

            ys, xs = np.where(component)
            if xs.size == 0:
                continue

            span_y = float(np.ptp(ys))
            span_x = float(np.ptp(xs))

            if span_y < min_span:
                continue

            # Reject tiny squat blobs. A curved lane may have large x-span, so
            # do not require it to be nearly vertical.
            if max(span_x, span_y) < min_span:
                continue

            border_touch = int(
                np.count_nonzero(component[:, :3])
                + np.count_nonzero(component[:, max(0, w - 3):])
            )
            border_frac = float(border_touch) / max(1.0, float(area))

            # Border contact is suspicious, but do not reject a genuine lane
            # just because it reaches the lower-left corner.
            border_penalty = 350.0 * max(
                0.0, border_frac - max_border_frac
            )

            road_overlap = 0
            if road_near is not None:
                road_overlap = int(
                    np.count_nonzero(component & road_near)
                )

            bottom_reach = float(np.max(ys)) / max(1.0, float(h - 1))

            # Dominant factors are vertical support + actual component area.
            # Road support is intentionally only a bonus.
            score = (
                10.0 * span_y
                + 2.0 * float(area)
                + 0.8 * span_x
                + 0.8 * float(road_overlap)
                + bottom_bonus_scale * bottom_reach
                - border_penalty
            )

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

        selected = np.zeros_like(yellow_mask, dtype=np.uint8)

        if best_label is not None:
            # Use the ORIGINAL threshold pixels belonging to/near the selected
            # connected component so morphology does not artificially widen it.
            selected_component = labels == best_label

            # Slight dilation is only for matching original yellow pixels.
            support = cv2.dilate(
                selected_component.astype(np.uint8) * 255,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            selected[
                (support > 0) & (yellow_mask > 0)
            ] = 255

            # If original pixels are fragmented too severely, retain the closed
            # component itself as a fallback.
            if np.count_nonzero(selected) < min_area:
                selected[selected_component] = 255

        return selected


    def _yellow_mask(self, roi_bgr):
        p = lambda name: self.get_parameter(name).value

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        h0 = int(clamp(float(p('yellow_hue_min')), 0.0, 179.0))
        h1 = int(clamp(float(p('yellow_hue_max')), 0.0, 179.0))
        s0 = int(clamp(float(p('yellow_saturation_min')), 0.0, 255.0))
        v0 = int(clamp(float(p('yellow_value_min')), 0.0, 255.0))

        mask = cv2.inRange(
            hsv,
            np.array([h0, s0, v0], dtype=np.uint8),
            np.array([h1, 255, 255], dtype=np.uint8),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        return mask

    # ==================================================================
    # LEFT yellow boundary: direct scanline tracker
    # ==================================================================
    def _track_yellow(self, mask, roi_top, full_width, road_mask=None):
        """Find the longest smooth yellow path across all visible scan rows.

        V20 selected one connected component first. That was better than V19,
        but on real hardware the painted stripe is sometimes broken into
        several components by lighting, glare and compression. It could then
        jump between fragments or disappear.

        V21 does not require one connected component. It builds candidate
        yellow runs on every scan row and uses dynamic programming to choose
        one globally continuous path. A small temporal bonus favours the path
        near the previous frame, but current image evidence dominates.
        """
        if mask is None or mask.size == 0:
            return self._maybe_hold_last_yellow()

        p = lambda name: self.get_parameter(name).value

        step = max(1, int(p('scan_row_step')))
        min_run = max(1, int(p('yellow_min_run_px')))
        max_run = max(min_run, int(p('yellow_max_run_px')))

        max_gap_rows = max(1, int(p('yellow_path_max_gap_rows')))
        max_jump = max(8.0, float(p('yellow_path_max_jump_px')))
        jump_penalty = max(0.0, float(p('yellow_path_jump_penalty')))
        gap_penalty = max(0.0, float(p('yellow_path_gap_penalty')))
        road_bonus_value = max(0.0, float(p('yellow_path_road_bonus')))
        temporal_bonus_value = max(
            0.0, float(p('yellow_path_temporal_bonus'))
        )
        temporal_scale = max(
            1.0, float(p('yellow_path_temporal_scale_px'))
        )
        min_score = float(p('yellow_path_min_score'))

        h, w = mask.shape[:2]

        # Road proximity is a BONUS only. It must never delete the real yellow
        # line, because the bright paint can sit slightly outside the dark-road
        # segmentation on a curve.
        road_near = None
        if road_mask is not None:
            dilate_px = max(
                1, int(self.get_parameter('yellow_road_dilate_px').value)
            )
            k = 2 * dilate_px + 1
            road_near = cv2.dilate(
                road_mask,
                np.ones((k, k), dtype=np.uint8),
                iterations=1,
            ) > 0

        # Previous-frame interpolator is also only a soft preference.
        previous_interp = None
        now = time.monotonic()
        if (
            self._last_left_points
            and self._last_left_points_time is not None
            and now - self._last_left_points_time <= 0.50
        ):
            previous_interp = self._build_interpolator(
                self._last_left_points
            )

        # Candidate structure per row:
        # {
        #   'y_roi', 'y_full',
        #   'candidates': [{'x','width','base'} ...]
        # }
        rows = []

        for y in range(0, h, step):
            fg = np.flatnonzero(mask[y] > 0)
            if fg.size == 0:
                continue

            split_at = np.where(np.diff(fg) > 1)[0] + 1
            runs = np.split(fg, split_at)

            candidates = []

            for run in runs:
                width = len(run)
                if width < min_run or width > max_run:
                    continue

                x = 0.5 * (
                    float(run[0]) + float(run[-1])
                )
                y_full = float(y + roi_top)

                # Every supported row is valuable. Slightly favour a run with
                # several pixels over a one-pixel speck.
                base = 2.2 + min(float(width), 16.0) * 0.08

                if road_near is not None:
                    xi = int(round(x))
                    if (
                        0 <= y < road_near.shape[0]
                        and 0 <= xi < road_near.shape[1]
                        and bool(road_near[y, xi])
                    ):
                        base += road_bonus_value

                if previous_interp is not None:
                    previous_x = previous_interp(y_full)
                    if previous_x is not None:
                        distance = abs(float(x) - float(previous_x))
                        base += temporal_bonus_value * math.exp(
                            -distance / temporal_scale
                        )

                # Border detections are possible on a real curve, so penalize
                # them softly rather than rejecting them.
                border_distance = min(x, float(w - 1) - x)
                if border_distance < 5.0:
                    base -= 1.5

                candidates.append({
                    'x': float(x),
                    'width': float(width),
                    'base': float(base),
                })

            if candidates:
                rows.append({
                    'y_roi': int(y),
                    'y_full': float(y + roi_top),
                    'candidates': candidates,
                })

        if not rows:
            return self._maybe_hold_last_yellow()

        # Dynamic programming.
        # Each candidate gets the best path score ending at that candidate.
        states = []

        for row_index, row in enumerate(rows):
            row_states = []

            for cand_index, cand in enumerate(row['candidates']):
                best_score = float(cand['base'])
                best_prev = None

                # Connect to candidates from recent earlier rows.
                for prev_row_index in range(
                    max(0, row_index - max_gap_rows),
                    row_index,
                ):
                    previous_row = rows[prev_row_index]
                    row_gap_px = (
                        row['y_roi'] - previous_row['y_roi']
                    )
                    row_gap_units = max(
                        1.0, float(row_gap_px) / float(step)
                    )

                    # Curves legitimately move more over a bigger missing gap.
                    allowed_jump = max_jump * (
                        1.0
                        + 0.18 * min(row_gap_units - 1.0, 6.0)
                    )

                    for prev_cand_index, prev_state in enumerate(
                        states[prev_row_index]
                    ):
                        dx = abs(
                            float(cand['x'])
                            - float(
                                previous_row['candidates'][
                                    prev_cand_index
                                ]['x']
                            )
                        )

                        if dx > allowed_jump:
                            continue

                        transition_penalty = (
                            jump_penalty * dx
                            + gap_penalty
                            * max(0.0, row_gap_units - 1.0)
                        )

                        score = (
                            float(prev_state['score'])
                            + float(cand['base'])
                            - transition_penalty
                        )

                        if score > best_score:
                            best_score = score
                            best_prev = (
                                prev_row_index,
                                prev_cand_index,
                            )

                row_states.append({
                    'score': float(best_score),
                    'prev': best_prev,
                })

            states.append(row_states)

        # Best endpoint can be anywhere. Reward vertical span implicitly through
        # the accumulated per-row score.
        best_endpoint = None
        best_score = None

        for row_index, row_states in enumerate(states):
            for cand_index, state in enumerate(row_states):
                score = float(state['score'])
                if best_score is None or score > best_score:
                    best_score = score
                    best_endpoint = (row_index, cand_index)

        if (
            best_endpoint is None
            or best_score is None
            or best_score < min_score
        ):
            return self._maybe_hold_last_yellow()

        # Backtrack selected path.
        selected = []
        cursor = best_endpoint

        while cursor is not None:
            row_index, cand_index = cursor
            row = rows[row_index]
            cand = row['candidates'][cand_index]

            selected.append(
                (
                    float(cand['x']),
                    float(row['y_full']),
                )
            )

            cursor = states[row_index][cand_index]['prev']

        # Backtracking gives lower/final row -> previous rows in reverse graph
        # order. Sort explicitly by y near->far for downstream consistency.
        selected = sorted(
            selected,
            key=lambda item: item[1],
            reverse=True,
        )

        if not self._points_valid(selected):
            return self._maybe_hold_last_yellow()

        # Reject isolated x spikes, but preserve genuine smooth curve motion.
        if len(selected) >= 7:
            xs = np.asarray(
                [item[0] for item in selected],
                dtype=float,
            )
            ys = np.asarray(
                [item[1] for item in selected],
                dtype=float,
            )

            med = median_filter_1d(xs, radius=2)
            residual = np.abs(xs - med)
            threshold = max(
                10.0,
                3.5 * float(np.median(residual) + 1.0),
            )

            keep = residual <= threshold

            if int(np.count_nonzero(keep)) >= 6:
                selected = [
                    (float(x), float(y))
                    for x, y, use in zip(xs, ys, keep)
                    if bool(use)
                ]

        selected = self._smooth_points(selected)

        if self._points_valid(selected):
            self._previous_left_x = float(selected[0][0])
            self._last_left_points = list(selected)
            self._last_left_points_time = now
            return selected

        return self._maybe_hold_last_yellow()

    def _maybe_hold_last_yellow(self):
        """Bridge only a very short one/two-frame segmentation dropout.

        The held path is for perception continuity; it is deliberately bounded
        to a fraction of a second so a stale lane cannot persist.
        """
        hold_sec = max(
            0.0,
            float(
                self.get_parameter(
                    'yellow_hold_last_sec'
                ).value
            ),
        )

        if (
            hold_sec > 0.0
            and self._last_left_points
            and self._last_left_points_time is not None
            and time.monotonic() - self._last_left_points_time <= hold_sec
        ):
            return list(self._last_left_points)

        self._previous_left_x = None
        return []

    # ==================================================================
    # RIGHT boundary: actual dark-road -> bright/white transition
    # ==================================================================
    def _track_right_edge(
        self,
        frame,
        roi_top,
        left_points,
        road_mask,
    ):
        """Track the RIGHT edge of the road-connected asphalt component.

        This is substantially safer than V17's unrestricted gradient search:
        a wall or shelf can have a strong dark->bright edge, but it is not the
        boundary of the asphalt component connected to the vehicle.
        """
        if road_mask is None:
            self._previous_right_x = None
            return []

        p = lambda name: self.get_parameter(name).value

        roi = frame[roi_top:]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape[:2]
        component = road_mask > 0

        step = max(1, int(p('scan_row_step')))
        border_excl = max(3, int(p('right_border_exclusion_px')))
        x_lo_global = int(
            clamp(float(p('right_search_min_x_frac')), 0.0, 0.95) * w
        )
        x_hi_global = int(
            clamp(float(p('right_search_max_x_frac')), 0.40, 1.0) * w
        )
        x_hi_global = min(w - 1 - border_excl, x_hi_global)

        ignore_top = int(
            clamp(
                float(p('right_ignore_top_frac_of_roi')),
                0.0,
                0.50,
            ) * h
        )

        min_left_support = max(
            4, int(p('right_min_road_pixels_left'))
        )
        contrast_min = max(0.0, float(p('right_contrast_min')))
        bright_min = max(0.0, float(p('right_bright_min')))
        yellow_margin = max(0, int(p('right_of_yellow_margin_px')))
        max_jump = max(8.0, float(p('right_max_row_jump_px')))

        left_interp = self._build_interpolator(left_points)

        raw_points = []

        for y in range(h - 1, ignore_top - 1, -step):
            xs = np.flatnonzero(component[y])
            if xs.size == 0:
                continue

            xs = xs[
                (xs >= x_lo_global) & (xs <= x_hi_global)
            ]
            if xs.size == 0:
                continue

            # Rightmost asphalt point is the road->white/curb boundary.
            x = int(xs[-1])

            y_full = float(y + roi_top)

            yellow_x = (
                left_interp(y_full)
                if left_interp is not None
                else None
            )
            if yellow_x is not None and x <= yellow_x + yellow_margin:
                continue

            # Reject tiny dark islands.
            l0 = max(0, x - 24)
            road_support = int(
                np.count_nonzero(component[y, l0:x + 1])
            )
            if road_support < min_left_support:
                continue

            # Verify that the area immediately outside the asphalt is actually
            # brighter. This identifies the white/pale track boundary.
            left0 = max(0, x - 10)
            left1 = max(1, x - 2)
            right0 = min(w - 1, x + 2)
            right1 = min(w, x + 12)

            if left1 <= left0 or right1 <= right0:
                continue

            inside_mean = float(
                np.mean(gray[y, left0:left1])
            )
            outside_mean = float(
                np.mean(gray[y, right0:right1])
            )

            if outside_mean < bright_min:
                continue
            if outside_mean - inside_mean < contrast_min:
                continue

            raw_points.append((float(x), y_full))

        if len(raw_points) < max(
            5, int(p('right_min_support_rows'))
        ):
            self._previous_right_x = None
            return []

        # Keep longest smooth chain. Start from bottom so the road edge nearest
        # QCar anchors the tracking; never let a horizon/wall edge seed it.
        best_chain = []
        chain = []
        prev_x = None
        prev_y = None

        for x, y in raw_points:
            if prev_x is None:
                chain = [(x, y)]
            else:
                row_gap = max(1.0, float(prev_y - y) / float(step))
                allowed = max_jump * (
                    1.0 + 0.15 * min(row_gap - 1.0, 5.0)
                )

                if abs(float(x) - float(prev_x)) <= allowed:
                    chain.append((x, y))
                else:
                    if len(chain) > len(best_chain):
                        best_chain = chain
                    chain = [(x, y)]

            prev_x = x
            prev_y = y

        if len(chain) > len(best_chain):
            best_chain = chain

        points = self._smooth_points(best_chain)

        if self._points_valid(points):
            self._previous_right_x = float(points[0][0])
            return points

        self._previous_right_x = None
        return []

    def _right_corridor_supported(self, left_points, right_points, frame_height, roi_top):
        """Check that the right trace really pairs with yellow over a near-field span.

        Reject short false positives at the image horizon (the red segment in
        the supplied screenshot). Reject traces far from the drivable road.
        This test does not manufacture any boundary samples.
        """
        if not left_points or not right_points:
            return False
        left_support = self._support(left_points)
        right_support = self._support(right_points)
        roi_height = max(1., float(frame_height - roi_top))
        overlap = self._support_overlap(
            left_support, right_support,
            min_span=max(float(self.get_parameter('minimum_support_span_px').value),
                         float(self.get_parameter('right_min_overlap_roi_frac').value) * roi_height))
        if overlap is None:
            return False
        # A stray bright horizon edge is too far from the lane nearest the car.
        max_near_gap = float(self.get_parameter('right_max_near_gap_roi_frac').value) * roi_height
        if float(left_support[1]) - float(right_support[1]) > max_near_gap:
            return False
        left_at = self._build_interpolator(left_points)
        right_at = self._build_interpolator(right_points)
        if left_at is None or right_at is None:
            return False
        widths = []
        for y in np.linspace(overlap[0], overlap[1], 15):
            lx, rx = left_at(y), right_at(y)
            if lx is not None and rx is not None:
                widths.append(rx - lx)
        min_width = max(10., float(self.get_parameter('right_of_yellow_margin_px').value))
        return bool(len(widths) >= 8 and np.mean(np.asarray(widths) > min_width) >= 0.85)

    def _track_right_white_scanlines(self, white_mask, roi_top, left_points, road_mask):
        """Fallback when a curved/broken Gazebo white stripe isn't one component.

        Uses only observed bright runs supported by black asphalt on their
        left. Traces a single contiguous near-to-far chain; no extrapolation.
        """
        h, w = white_mask.shape[:2]
        p = lambda name: self.get_parameter(name).value
        left_at = self._build_interpolator(left_points)
        if left_at is None:
            return []
        step = max(1, int(p('scan_row_step')))
        margin = max(5, int(p('right_of_yellow_margin_px')))
        max_width = max(5, int(float(p('right_white_max_band_width_frac')) * w))
        min_road = float(p('right_white_min_asphalt_support'))
        rows = []
        for row in range(h-1, -1, -step):
            left_x = left_at(row + roi_top)
            if left_x is None:
                continue
            xs = np.flatnonzero(white_mask[row] > 0)
            if xs.size == 0:
                continue
            runs = np.split(xs, np.where(np.diff(xs) > 1)[0]+1)
            choices = []
            for run in runs:
                x = int(run[0])
                if len(run) < 2 or len(run) > max_width:
                    continue
                if x < 4 or x >= w-5 or x < left_x + margin:
                    continue
                lo, hi = max(0,x-18), max(1,x-3)
                if hi <= lo or np.mean(road_mask[row,lo:hi] > 0) < min_road:
                    continue
                choices.append(float(x))
            if choices:
                rows.append((row,choices))
        if not rows:
            return []
        max_gap = max(2, int(p('max_missing_rows')))
        max_jump = max(10., float(p('right_max_row_jump_px')))
        chains = []
        # Only near-field evidence can seed a white-boundary track; never
        # start tracking on an isolated bright patch at the far horizon.
        left_near = max(y for _,y in left_points)
        seeds = [(r,choices) for r,choices in rows if r + roi_top >= left_near - 0.2*h]
        for seed_row, seed_choices in seeds[:12]:
            for seed_x in seed_choices[:6]:
                chain=[(seed_x, float(seed_row+roi_top))]
                prev_row,prev_x=seed_row,seed_x
                for row,choices in rows:
                    if row >= prev_row:
                        continue
                    gap = (prev_row-row)/step
                    if gap > max_gap:
                        break
                    closest = min(choices,key=lambda x:abs(x-prev_x))
                    if abs(closest-prev_x) > max_jump*(1+0.1*gap):
                        continue
                    chain.append((float(closest),float(row+roi_top)))
                    prev_row,prev_x = row,closest
                if self._points_valid(chain):
                    span = chain[0][1] - chain[-1][1]
                    if span >= float(p('right_white_min_span_frac'))*h:
                        chains.append((span,len(chain),chain))
        if not chains:
            return []
        result=self._smooth_points(max(chains,key=lambda c:(c[0],c[1]))[2])
        return result if self._points_valid(result) else []

    def _track_right_white_stripe(self, frame, roi_top, left_points, road_mask):
        """Gazebo-only fallback: the observed INNER edge of white paint.

        No polynomial extrapolation, no synthetic middle lane, and no search
        outside the visible camera rows. This is used ONLY when V23's usual
        bottom-connected asphalt/curb edge detector found no usable track.
        The selected white stripe must be spatially to the right of yellow,
        vertically supported, bright/low-saturation and asphalt-adjacent.
        """
        p = lambda name: self.get_parameter(name).value
        roi = frame[roi_top:]
        h, w = roi.shape[:2]
        if h < 16 or w < 32 or not left_points or road_mask is None:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv,
            np.array([0, 0, int(p('right_white_value_min'))], dtype=np.uint8),
            np.array([179, int(p('right_white_saturation_max')), 255], dtype=np.uint8),
        )
        # Join short interruptions in a line, but don't bridge large gaps.
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8),
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            white, connectivity=8,
        )
        if n <= 1:
            return []

        left_interp = self._build_interpolator(left_points)
        if left_interp is None:
            return []
        min_span = max(
            float(p('minimum_support_span_px')),
            float(p('right_white_min_span_frac')) * h,
        )
        min_area = max(10, int(p('right_white_min_area_px')))
        max_width = max(4.0, float(p('right_white_max_band_width_frac')) * w)
        yellow_margin = max(0, int(p('right_of_yellow_margin_px')))
        min_asphalt = float(p('right_white_min_asphalt_support'))
        min_rows = max(5, int(p('right_min_support_rows')))
        step = max(1, int(p('scan_row_step')))
        candidates = []

        for label in range(1, n):
            x0, y0, bw, bh, area = [int(v) for v in stats[label]]
            if area < min_area or bh < min_span:
                continue
            # An entire bright wall/sky or a wide curb isn't a thin stripe.
            if bw > 0.80 * w:
                continue
            component = labels == label
            samples = []
            for row in range(y0 + bh - 1, y0 - 1, -step):
                indices = np.flatnonzero(component[row])
                if indices.size < 2:
                    continue
                # The drivable corridor ends where the WHITE stripe begins.
                x = int(indices[0])
                # Restrict to the inside of the visible image, not the border.
                if x < 3 or x >= w - 3:
                    continue
                # Skip very broad bright regions (walls, sky, curb surfaces).
                band_width = int(indices[-1]) - x + 1
                if band_width > max_width:
                    continue
                y_full = float(row + roi_top)
                left_x = left_interp(y_full)
                if left_x is None or x <= left_x + yellow_margin:
                    continue
                # A real painted right stripe borders dark road on its left.
                # This gate filters bright walls not touching the drivable lane.
                lo = max(0, x - 15)
                hi = max(lo + 1, x - 3)
                asphalt_fraction = float(
                    np.mean(road_mask[row, lo:hi] > 0)
                )
                if asphalt_fraction < min_asphalt:
                    continue
                samples.append((float(x), y_full))
            if len(samples) < min_rows:
                continue
            samples = self._smooth_points(samples)
            if not self._points_valid(samples):
                continue
            # Prefer longer observed vertical support, then number of rows.
            span = samples[0][1] - samples[-1][1]
            candidates.append((span, len(samples), samples))

        if not candidates:
            samples = self._track_right_white_scanlines(
                white, roi_top, left_points, road_mask)
            if samples:
                self._previous_right_x = float(samples[0][0])
            return samples
        samples = max(candidates, key=lambda item: (item[0], item[1]))[2]
        self._previous_right_x = float(samples[0][0])
        return samples

    def _choose_valid_right_boundary(self, frame, roi_top, left_points, road_mask):
        """Select a measured right lane edge that yields a *usable* forward path.

        A vertical curb or unrelated bright edge can pass row-overlap tests,
        yet project to near-zero forward distance. Such a detection must not
        be reported as FULL or reach the control loop. Evaluate both the V23
        asphalt-edge candidate and the white-stripe fallback BEFORE choosing.
        Neither branch invents pixels/waypoints outside their measured rows.
        """
        height, width = frame.shape[:2]
        self._right_source = 'none'
        self._right_span_m = 0.0
        self._right_rejection_details = ''
        if not left_points or road_mask is None:
            self._right_rejection_details = 'left boundary / connected road missing'
            return []

        p = lambda name: self.get_parameter(name).value
        min_span = max(0.0, float(p('min_path_forward_span_m')))
        min_points = max(3, int(p('min_path_points')))

        # Preserve the original V23 road-edge implementation, but do not
        # accept its output merely because the image-row overlap looks valid.
        candidates = [('asphalt-edge', self._track_right_edge(
            frame, roi_top, left_points, road_mask))]
        if bool(p('right_white_fallback')):
            candidates.append(('white-stripe', self._track_right_white_stripe(
                frame, roi_top, left_points, road_mask)))

        valid = []
        rejected = []
        for source, right in candidates:
            if not self._right_corridor_supported(
                    left_points, right, height, roi_top):
                rejected.append(source + ': poor two-edge image support')
                continue
            centre, _, status = self._centreline_points(
                left_points, right, width, height, roi_top)
            if status != LANE_STATUS_FULL or len(centre) < min_points:
                rejected.append(source + ': insufficient matched centerline')
                continue
            # The same perspective conversion used by actual path publication:
            # do not judge geometric reach using image-row span alone.
            state = self._metric_state(centre, width, height, roi_top)
            path = self._current_metric_path
            if state is None or path is None or len(path) < min_points:
                rejected.append(source + ': cannot build metric centerline')
                continue
            span = float(path[-1, 0] - path[0, 0])
            if not math.isfinite(span) or span < min_span:
                rejected.append(source + ': metric forward span '
                                + f'{span:.3f} m < {min_span:.3f} m')
                continue
            valid.append((span, source, right))

        # _metric_state temporarily writes _current_metric_path while evaluating
        # candidates. The caller recomputes it only for the accepted boundary.
        self._current_metric_path = None
        if not valid:
            self._right_rejection_details = '; '.join(rejected)
            return []

        # Prefer the observed inner white paint in Gazebo when it gives a
        # comparable span; use V23's dark-road edge in environments where it
        # gives demonstrably better geometric support (e.g. physical QCar).
        best_span = max(entry[0] for entry in valid)
        nearby = [entry for entry in valid if entry[0] >= 0.8 * best_span]
        selected = next((entry for entry in nearby if entry[1] == 'white-stripe'),
                        max(valid, key=lambda entry: entry[0]))
        self._right_span_m, self._right_source, right = selected
        return right

    # ==================================================================
    # Point utilities
    # ==================================================================
    def _smooth_points(self, points):
        if len(points) < 3:
            return points

        # points are bottom -> top
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)

        xs = median_filter_1d(xs, radius=2)

        # Small moving average after median: smooth without changing the
        # actual curve into a globally fitted parabola.
        if len(xs) >= 5:
            smooth = xs.copy()
            for i in range(len(xs)):
                lo = max(0, i - 2)
                hi = min(len(xs), i + 3)
                smooth[i] = float(np.mean(xs[lo:hi]))
            xs = smooth

        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def _points_valid(self, points):
        p = lambda name: self.get_parameter(name).value

        if len(points) < max(4, int(p('minimum_boundary_points'))):
            return False

        ys = np.asarray([point[1] for point in points], dtype=float)
        if float(np.ptp(ys)) < float(p('minimum_support_span_px')):
            return False

        return True

    @staticmethod
    def _build_interpolator(points):
        if points is None or len(points) < 2:
            return None

        arr = np.asarray(points, dtype=float)
        xs = arr[:, 0]
        ys = arr[:, 1]

        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]

        y_min = float(ys[0])
        y_max = float(ys[-1])

        def interp(y):
            y = float(y)
            if y < y_min or y > y_max:
                return None
            return float(np.interp(y, ys, xs))

        return interp

    @staticmethod
    def _support(points):
        if not points:
            return None
        ys = [p[1] for p in points]
        return (float(min(ys)), float(max(ys)))

    @staticmethod
    def _support_overlap(a, b, min_span=12.0):
        if a is None or b is None:
            return None
        lo = max(float(a[0]), float(b[0]))
        hi = min(float(a[1]), float(b[1]))
        if hi - lo < min_span:
            return None
        return lo, hi

    # ==================================================================
    # Lane width
    # ==================================================================
    def _default_width(self, y, img_w, img_h, roi_top):
        p = lambda name: self.get_parameter(name).value

        bottom = (
            self._learned_width_near
            if self._learned_width_near is not None
            else float(img_w) * float(p('default_lane_width_bottom_frac'))
        )
        top = (
            self._learned_width_far
            if self._learned_width_far is not None
            else float(img_w) * float(p('default_lane_width_top_frac'))
        )

        frac = (
            (float(img_h - 1) - float(y))
            / max(1.0, float(img_h - 1 - roi_top))
        )
        frac = clamp(frac, 0.0, 1.0)

        return bottom + frac * (top - bottom)

    def _update_width(self, left_points, right_points, img_w, img_h, roi_top):
        overlap = self._support_overlap(
            self._support(left_points),
            self._support(right_points),
            min_span=20.0,
        )
        if overlap is None:
            return

        left_i = self._build_interpolator(left_points)
        right_i = self._build_interpolator(right_points)
        if left_i is None or right_i is None:
            return

        y_far, y_near = overlap
        near_width = right_i(y_near) - left_i(y_near)
        far_width = right_i(y_far) - left_i(y_far)

        if not (
            near_width is not None
            and far_width is not None
            and near_width > 20.0
            and far_width > 12.0
        ):
            return

        alpha = clamp(
            float(self.get_parameter('lane_width_alpha').value),
            0.0,
            1.0,
        )

        if 0.10 * img_w <= near_width <= 0.95 * img_w:
            if self._learned_width_near is None:
                self._learned_width_near = float(near_width)
            else:
                self._learned_width_near = (
                    (1.0 - alpha) * self._learned_width_near
                    + alpha * float(near_width)
                )

        if 0.05 * img_w <= far_width <= 0.75 * img_w:
            if self._learned_width_far is None:
                self._learned_width_far = float(far_width)
            else:
                self._learned_width_far = (
                    (1.0 - alpha) * self._learned_width_far
                    + alpha * float(far_width)
                )

    # ==================================================================
    # Centreline from actual sampled points
    # ==================================================================
    def _centreline_points(
        self,
        left_points,
        right_points,
        img_w,
        img_h,
        roi_top,
    ):
        left_s = self._support(left_points)
        right_s = self._support(right_points)

        left_i = self._build_interpolator(left_points)
        right_i = self._build_interpolator(right_points)

        if left_i is not None and right_i is not None:
            overlap = self._support_overlap(left_s, right_s, min_span=16.0)
            if overlap is not None:
                y_far, y_near = overlap
                ys = np.linspace(y_near, y_far, 36)
                points = []

                for y in ys:
                    lx = left_i(y)
                    rx = right_i(y)
                    if lx is None or rx is None:
                        continue
                    if rx <= lx + 10.0:
                        continue

                    points.append(
                        (
                            0.5 * (lx + rx),
                            float(y),
                            float(rx - lx),
                        )
                    )

                if len(points) >= 8:
                    return points, 'both boundaries', LANE_STATUS_FULL

        # In driving mode there is no mathematically justified centreline
        # without both boundary samples at the same image rows. Do not turn
        # a guessed lane width into a magenta line that can float across bends.
        if bool(self.get_parameter('require_both_boundaries').value):
            if left_i is not None or right_i is not None:
                return [], 'ONE BOUNDARY: centreline unavailable', LANE_STATUS_PARTIAL
            return [], 'no boundaries', LANE_STATUS_LOST

        # One-boundary fallback. Keep it simple and local; no temporal curve
        # propagation and no global polynomial.
        if left_i is not None and left_s is not None:
            y_far, y_near = left_s
            ys = np.linspace(y_near, y_far, 30)
            points = []

            for y in ys:
                lx = left_i(y)
                if lx is None:
                    continue
                width = self._default_width(
                    y, img_w, img_h, roi_top
                )
                points.append(
                    (
                        lx + 0.5 * width,
                        float(y),
                        float(width),
                    )
                )

            if len(points) >= 8:
                return points, 'yellow only + half width', LANE_STATUS_PARTIAL

        if right_i is not None and right_s is not None:
            y_far, y_near = right_s
            ys = np.linspace(y_near, y_far, 30)
            points = []

            for y in ys:
                rx = right_i(y)
                if rx is None:
                    continue
                width = self._default_width(
                    y, img_w, img_h, roi_top
                )
                points.append(
                    (
                        rx - 0.5 * width,
                        float(y),
                        float(width),
                    )
                )

            if len(points) >= 8:
                return points, 'right edge only - half width', LANE_STATUS_PARTIAL

        return [], 'none', LANE_STATUS_LOST

    # ==================================================================
    # Metric lane state
    # ==================================================================
    def _metric_state(self, centre_points, img_w, img_h, roi_top):
        self._current_metric_path = None
        if len(centre_points) < 6:
            return None

        p = lambda name: self.get_parameter(name).value

        lane_width_m = max(0.05, float(p('lane_width_m')))
        forward_range = max(0.10, float(p('visible_forward_range_m')))

        # centre points are near -> far
        xs = np.asarray([item[0] for item in centre_points], dtype=float)
        ys = np.asarray([item[1] for item in centre_points], dtype=float)
        widths = np.asarray(
            [max(10.0, item[2]) for item in centre_points],
            dtype=float,
        )

        lateral = (
            (0.5 * float(img_w) - xs)
            * lane_width_m
            / widths
        )

        y_bottom = float(img_h - 1)

        if self._camera_fx is not None:
            depth = self._camera_fx * lane_width_m / widths
            forward = depth - float(depth[0])
            forward = np.maximum.accumulate(
                np.maximum(forward, 0.0)
            )
        else:
            forward = (
                (y_bottom - ys)
                / max(1.0, y_bottom - float(roi_top))
                * forward_range
            )

        valid = (
            np.isfinite(forward)
            & np.isfinite(lateral)
            & (forward >= 0.0)
            & (forward <= 1.25 * forward_range)
        )

        forward = forward[valid]
        lateral = lateral[valid]

        if len(forward) < 6:
            return None

        # Remove repeated forward values.
        unique = np.r_[True, np.diff(forward) > 1.0e-4]
        forward = forward[unique]
        lateral = lateral[unique]

        if len(forward) < 6:
            return None

        # CTE: use the nearest actually observed centre point, not an
        # extrapolated polynomial intercept.
        cte = float(lateral[0])

        # Heading: local line fit over the nearest heading horizon.
        heading_dist = max(0.08, float(p('heading_fit_distance_m')))
        heading_mask = forward <= (
            float(forward[0]) + heading_dist
        )

        if int(np.count_nonzero(heading_mask)) < 4:
            heading_mask = np.arange(len(forward)) < min(8, len(forward))

        try:
            line = np.polyfit(
                forward[heading_mask],
                lateral[heading_mask],
                1,
            )
        except Exception:
            return None

        path_heading = math.atan(float(line[0]))
        heading_error = -path_heading

        # Curvature: quadratic fit over visible near horizon, but explicitly
        # classify straight data before accepting curvature.
        curve_dist = max(0.20, float(p('curvature_fit_distance_m')))
        curve_mask = forward <= (
            float(forward[0]) + curve_dist
        )

        if int(np.count_nonzero(curve_mask)) < 6:
            curve_mask = np.ones(len(forward), dtype=bool)

        fs = forward[curve_mask]
        ls = lateral[curve_mask]

        try:
            line2 = np.polyfit(fs, ls, 1)
            line_pred = np.polyval(line2, fs)
            line_rmse = float(
                np.sqrt(np.mean((ls - line_pred) ** 2))
            )

            quad = np.polyfit(fs, ls, 2)
            a = float(quad[0])
            b = float(quad[1])

            s0 = float(np.min(fs))
            s1 = float(np.max(fs))

            h0 = math.atan(2.0 * a * s0 + b)
            h1 = math.atan(2.0 * a * s1 + b)
            heading_change = abs(
                normalize_angle(h1 - h0)
            )

            if (
                line_rmse <= float(p('straight_rmse_m'))
                and heading_change
                <= float(p('straight_heading_change_rad'))
            ):
                curvature = 0.0
            else:
                s_eval = 0.5 * (s0 + s1)
                slope = 2.0 * a * s_eval + b
                curvature = (
                    2.0 * a
                    / math.pow(1.0 + slope * slope, 1.5)
                )
        except Exception:
            curvature = 0.0

        if not all(
            math.isfinite(value)
            for value in (cte, heading_error, curvature)
        ):
            return None

        if abs(cte) > float(p('max_abs_cte_m')):
            return None
        if abs(heading_error) > float(p('max_abs_heading_rad')):
            return None
        if abs(curvature) > float(p('max_abs_curvature')):
            # A noisy curvature fit must not discard an otherwise valid CTE
            # and heading measurement. Curvature is not used for steering in
            # the current constant-speed PID/PP/Stanley hardware tests.
            curvature = 0.0

        # The exact camera-space points used to derive V23 tracking errors
        # are also supplied to the path publisher. No duplicate curve fitting,
        # extrapolation, or changes to the tracking logic.
        self._current_metric_path = np.column_stack((forward, lateral))
        return cte, heading_error, curvature

    def _filter_state(self, cte, heading, curvature):
        p = lambda name: self.get_parameter(name).value

        a_cte = clamp(float(p('cte_filter_alpha')), 0.0, 1.0)
        a_heading = clamp(
            float(p('heading_filter_alpha')), 0.0, 1.0
        )
        a_curve = clamp(
            float(p('curvature_filter_alpha')), 0.0, 1.0
        )

        if self._filtered_cte is None:
            self._filtered_cte = float(cte)
        else:
            self._filtered_cte = (
                a_cte * float(cte)
                + (1.0 - a_cte) * self._filtered_cte
            )

        if self._filtered_heading is None:
            self._filtered_heading = float(heading)
        else:
            delta = normalize_angle(
                float(heading) - self._filtered_heading
            )
            self._filtered_heading = normalize_angle(
                self._filtered_heading + a_heading * delta
            )

        if self._filtered_curvature is None:
            self._filtered_curvature = float(curvature)
        else:
            self._filtered_curvature = (
                a_curve * float(curvature)
                + (1.0 - a_curve) * self._filtered_curvature
            )

        return (
            float(self._filtered_cte),
            float(self._filtered_heading),
            float(self._filtered_curvature),
        )

    # ==================================================================
    # Corridor mask
    # ==================================================================
    def _corridor_mask(
        self,
        roi_shape,
        left_points,
        right_points,
        roi_top,
    ):
        h, w = roi_shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)

        left_i = self._build_interpolator(left_points)
        right_i = self._build_interpolator(right_points)

        if left_i is None:
            return out

        left_s = self._support(left_points)
        right_s = self._support(right_points)

        # Always draw the yellow trace.
        for x, y_full in left_points:
            y = int(round(y_full - roi_top))
            xi = int(round(x))
            if 0 <= y < h and 0 <= xi < w:
                out[y, max(0, xi - 2):min(w, xi + 3)] = 255

        if right_i is None:
            return out

        overlap = self._support_overlap(left_s, right_s, min_span=8.0)
        if overlap is None:
            return out

        # Draw only the actual two corridor boundaries. Nothing left of yellow.
        y_far, y_near = overlap
        for y_full in np.arange(y_far, y_near + 1.0, 1.0):
            lx = left_i(y_full)
            rx = right_i(y_full)
            if lx is None or rx is None or rx <= lx:
                continue

            y = int(round(y_full - roi_top))
            if not (0 <= y < h):
                continue

            lxi = int(round(lx))
            rxi = int(round(rx))

            out[y, max(0, lxi - 2):min(w, lxi + 3)] = 255
            out[y, max(0, rxi - 2):min(w, rxi + 3)] = 255

        return out

    # ==================================================================
    # Main
    # ==================================================================
    def _process_frame(self, frame, frame_id):
        p = lambda name: self.get_parameter(name).value

        img_h, img_w = frame.shape[:2]
        roi_top = int(
            img_h * clamp(float(p('roi_top_frac')), 0.0, 0.90)
        )

        roi = frame[roi_top:]

        road_mask = self._road_component(roi)

        raw_yellow_mask = self._yellow_mask(roi)

        # V21 searches all yellow runs globally instead of forcing the whole
        # stripe to be one connected component.
        yellow_mask = raw_yellow_mask

        last_fresh_yellow_time = self._last_left_points_time
        left_points = self._track_yellow(
            yellow_mask,
            roi_top,
            img_w,
            road_mask=road_mask,
        )
        # _track_yellow can hold a past trace for 0.18 s. For physical
        # recovery, held pixels are NOT a fresh independent observation.
        # This is a read-only observation of the original V23 tracker.
        self._recovery_yellow_fresh_this_frame = bool(
            left_points and self._last_left_points_time is not None
            and self._last_left_points_time != last_fresh_yellow_time)

        right_points = self._choose_valid_right_boundary(
            frame, roi_top, left_points, road_mask)
        if left_points and not right_points and not bool(p('enable_single_boundary_recovery')):
            # With recovery enabled, report the outcome of reconstruction below
            # instead of warning merely because its trigger occurred.
            self._warn_path_blocked(
                'No usable right lane boundary: ' + self._right_rejection_details)

        self._update_width(
            left_points,
            right_points,
            img_w,
            img_h,
            roi_top,
        )

        centre_points, centre_source, status = self._centreline_points(
            left_points,
            right_points,
            img_w,
            img_h,
            roi_top,
        )

        state = self._metric_state(
            centre_points,
            img_w,
            img_h,
            roi_top,
        )

        # Isolated recovery: original FULL branch is unchanged except that we
        # remember the measured yellow-to-centre pixel offset AFTER its path
        # passes all original publication checks.
        if state is None:
            self._publish_status(status)
            if status == LANE_STATUS_PARTIAL:
                recovered = self._try_yellow_recovery(
                    left_points, right_points, img_w, img_h, roi_top)
                if recovered is not None:
                    centre_points, state = recovered
                    centre_source = ('RECOVERY: centre OUTSIDE camera view (see odom Path)'
                                     if len(centre_points)==0 else
                                     'RECOVERY: live yellow + ground-plane normal offset')
                else:
                    self._warn_path_blocked('only one supported lane boundary; recovery unavailable: ' + self._recovery_last_reason)
                    # The ORIGINAL V23 overlay is unchanged for FULL frames.
                    # For PARTIAL frames show the reason a new recovered path was rejected.
                    reason = str(self._recovery_last_reason)
                    if 'ground scale' in reason or 'FULL' in reason or 'cached' in reason:
                        code = 'NO FULL SCALE'
                    elif 'expired' in reason:
                        code = 'FULL SCALE EXPIRED'
                    elif 'HELD' in reason or 'fresh' in reason:
                        code = 'NO FRESH YELLOW'
                    elif 'curvature' in reason:
                        code = 'CURVATURE LIMIT'
                    elif 'fit' in reason or 'outlier' in reason:
                        code = 'YELLOW FIT REJECTED'
                    elif 'span' in reason or 'preview' in reason or 'overlap' in reason:
                        code = 'SHORT / NONOVERLAP'
                    elif 'odom' in reason:
                        code = 'ODOM STALE'
                    elif 'disagree' in reason:
                        code = 'PATH INCONSISTENT'
                    elif 'disabled' in reason:
                        code = 'FEATURE DISABLED'
                    else:
                        code = 'SEE DIAGNOSTICS'
                    centre_source = 'RECOVERY BLOCKED: ' + code
                    self._recovery_valid_pub.publish(Float64(data=0.0))
            else:
                self._recovery_valid_pub.publish(Float64(data=0.0))
        else:
            self._recovery_valid_pub.publish(Float64(data=0.0))
            cte, heading, curvature = self._filter_state(*state)
            self._publish_state(
                cte,
                heading,
                curvature,
                status,
            )
            self._last_valid_time = time.monotonic()
            if self._publish_reference_path(self._current_metric_path, status):
                self._learn_full_to_yellow_alignment(left_points)
                self._remember_full_yellow_offset(centre_points, left_points, img_w)

        right_edge_mask = np.zeros(
            yellow_mask.shape,
            dtype=np.uint8,
        )
        for x, y_full in right_points:
            y = int(round(y_full - roi_top))
            xi = int(round(x))
            if (
                0 <= y < right_edge_mask.shape[0]
                and 0 <= xi < right_edge_mask.shape[1]
            ):
                right_edge_mask[
                    y,
                    max(0, xi - 2):min(
                        right_edge_mask.shape[1], xi + 3
                    ),
                ] = 255

        corridor = self._corridor_mask(
            yellow_mask.shape,
            left_points,
            right_points,
            roi_top,
        )

        if bool(p('debug')):
            self._publish_debug(
                frame,
                roi_top,
                yellow_mask,
                right_edge_mask,
                road_mask,
                corridor,
                left_points,
                right_points,
                centre_points,
                centre_source,
                state,
                status,
                frame_id,
            )

    # ==================================================================
    # Bounded yellow-only recovery. NEVER called in normal FULL processing.
    # ==================================================================
    def _remember_full_yellow_offset(self, centre_points, left_points, img_width):
        """Record last validated FULL path; do NOT estimate camera pitch from width.

        The height and pitch are static parameters, not overwritten by changes
        in apparent lane width at sharp bends. Every accepted FULL frame renews
        our transition anchor, even if a pixel-offset profile is unlearnable.
        """
        odom_pose, odom_wall, _ = self._odom_snapshot()
        if self._latest_accepted_full_world_path is None or odom_pose is None or odom_wall is None:
            return
        self._recovery_full_world_path=np.asarray(
            self._latest_accepted_full_world_path,dtype=float).copy()
        self._recovery_full_image_width=int(img_width)
        self._recovery_full_time=time.monotonic()
        self._recovery_full_frame=str(odom_pose[3])
        self._recovery_consecutive=0
        self._recovery_travel_m=0.0
        self._recovery_prev_odom_xy=tuple(odom_pose[:2])
        self._recovery_prev_odom_wall_time=float(odom_wall)
        self._recovery_prev_world_path=None
        self._recovery_prev_world_time=None
        self._recovery_calibration_reject_reason=''

    def _recovery_diagnostic(self, accepted, reason, **more):
        self._recovery_last_reason = str(reason)
        now = time.monotonic()
        age = (None if self._recovery_full_time is None else
               max(0.0, now - self._recovery_full_time))
        _, odom_last, odom_count = self._odom_snapshot()
        odom_age = None if odom_last is None else max(0.0, now - odom_last)
        payload = {
            'accepted': bool(accepted), 'reason': str(reason),
            'last_full_age_sec': age,
            'odom_age_sec': odom_age,
            'odom_messages_seen': int(odom_count),
            'odom_freshness_limit_sec': float(self.get_parameter('odom_freshness_sec').value),
            'full_offset_cached': self._recovery_full_offset_profile is not None,
            'ground_scale_cached': self._recovery_full_world_path is not None,
            'last_calibration_rejection': self._recovery_calibration_reject_reason,
            'full_path_cached': self._recovery_full_world_path is not None,
            'recovery_enabled': bool(self.get_parameter('enable_single_boundary_recovery').value),
            'max_age_sec': min(0.30, float(self.get_parameter('recovery_max_age_sec').value)),
            'visual_max_age_sec': min(20.0,float(self.get_parameter('recovery_visual_max_age_sec').value)),
            'drive_max_age_sec': min(1.5,float(self.get_parameter('recovery_drive_max_age_sec').value)),
            'drive_max_travel_m': min(.45,float(self.get_parameter('recovery_drive_max_travel_m').value)),
        }
        payload.update(more)
        self._recovery_diagnostic_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _try_yellow_recovery(self, left_points, right_points, width, height, roi_top):
        """Continuous camera-derived candidate for PARTIAL yellow-only frames.

        A LAST measured FULL corridor supplies ONLY the row-dependent *scale*,
        never the current curve.  Current yellow pixels determine the current
        curvature every frame. The original FULL state estimator is untouched.
        Fail-closed for missing current observations, camera calibration,
        odometry, poor geometry, or excessive deviation between fresh frames.
        """
        p=lambda name:self.get_parameter(name).value
        basic={'yellow_points':len(left_points),'right_points':len(right_points)}
        def reject(reason,**details):
            self._recovery_consecutive=0
            self._recovery_diagnostic(False,reason,**basic,**details)
            return None
        if not bool(p('enable_single_boundary_recovery')):
            return reject('disabled: enable_single_boundary_recovery is false')
        if not left_points or right_points:
            return reject('not a yellow-only PARTIAL frame')
        if not getattr(self,'_recovery_yellow_fresh_this_frame',False):
            return reject('yellow trace was HELD from an old frame, not freshly detected')
        # No FULL-path age/width/anchor gate for publishing a newly observed
        # yellow-only reference. Keep the original recovery geometry algorithm.
        now=time.monotonic()
        age=(now-self._recovery_full_time) if self._recovery_full_time is not None else float('inf')
        odom_pose, odom_last, _ = self._odom_snapshot()
        if odom_pose is None or odom_last is None:
            return reject('odometry missing')
        odom_age = max(0.0, now - odom_last)
        if odom_age > float(p('odom_freshness_sec')):
            return reject('odometry stale: %.3fs (limit %.3fs)' %
                          (odom_age, float(p('odom_freshness_sec'))))
        if odom_pose[3]!=self._recovery_full_frame:
            return reject('odometry frame changed')
        if any(v is None for v in (self._camera_fx,self._camera_fy,self._camera_cx,self._camera_cy)):
            return reject('calibrated camera intrinsics missing')
        bx,by,byaw,frame=odom_pose
        # Track travel for diagnostics/drive authorization without withholding
        # the visual reference because of an additional odometry-jump gate.
        odom_step_m=0.0
        odom_step_dt_sec=0.0
        if self._recovery_prev_odom_xy is not None:
            prev_t=self._recovery_prev_odom_wall_time
            if prev_t is not None:
                odom_step_dt_sec=float(odom_last)-float(prev_t)
            odom_step_m=math.hypot(bx-self._recovery_prev_odom_xy[0],
                                   by-self._recovery_prev_odom_xy[1])
            if math.isfinite(odom_step_m):
                self._recovery_travel_m+=odom_step_m
        self._recovery_prev_odom_xy=(bx,by)
        self._recovery_prev_odom_wall_time=float(odom_last)
        raw,dbg,geometry=recover_ground_curve_normal_from_yellow(
            left_points,fx=float(self._camera_fx),fy=float(self._camera_fy),
            cx=float(self._camera_cx),cy=float(self._camera_cy),
            camera_height_m=float(p('ground_camera_height_m')),
            camera_pitch_rad=float(p('ground_camera_pitch_rad')),
            max_range_m=float(p('ground_max_recovery_range_m')),
            lane_width_m=float(p('lane_width_m')),
            fit_max_rmse_m=float(p('recovery_curve_fit_max_rmse_m')),
            min_inlier_fraction=float(p('recovery_curve_min_inlier_fraction')))
        if raw is None:
            return reject(geometry['reason'])
        # FULL observations calibrate the frame/scale of this metric curve;
        # use the same correction on PARTIAL, without changing V23 FULL paths.
        raw = self._align_yellow_recovery(raw)
        # A global straight-line fit can be ill-conditioned during a bend.
        # The recorded PID second turn repeatedly produced 0.8m straight
        # previews alternating with 0.16m curved observations. Do not let a
        # line-only estimate claim a longer preview than the trusted near
        # image support. Never extend or synthesize ground points.
        if geometry.get('fit_model') == 'line':
            seg=np.linalg.norm(np.diff(raw,axis=0),axis=1)
            arc=np.r_[0.,np.cumsum(seg)]
            trusted_preview_m=0.28
            if arc[-1] > trusted_preview_m:
                upto=int(np.searchsorted(arc,trusted_preview_m))
                tip=np.array([np.interp(trusted_preview_m,arc,raw[:,0]),
                              np.interp(trusted_preview_m,arc,raw[:,1])])
                raw=np.vstack((raw[:upto],tip))
        cleaned, info=sanitize_recovery_ground_arclength(
            raw,spacing_m=float(p('lane_path_spacing_m')),
            min_span_m=max(float(p('lane_path_min_span_m')),float(p('min_path_forward_span_m'))),
            max_curvature_m_inv=float(p('lane_path_max_curvature_m_inv')),
            max_observation_gap_m=float(p('lane_path_max_observation_gap_m')))
        if cleaned is None or len(cleaned)<max(8,int(p('min_path_points'))):
            return reject('recovery metric geometry rejected: '+str(info.get('reason')),
                          raw_span_m=geometry.get('raw_span_m'),
                          p95_curvature_m_inv=info.get('p95_curvature_m_inv'))
        state=state_from_recovery_ground(cleaned,float(p('heading_fit_distance_m')))
        if state is None:
            return reject('recovery metric state invalid')
        try:
            world=camera_centerline_to_world(
                cleaned[:,0],cleaned[:,1],bx,by,byaw,
                float(p('camera_x_from_base_m')),float(p('camera_y_from_base_m')),
                float(p('camera_yaw_from_base_rad')))
        except ValueError:
            return reject('camera-to-odom transform failed')
        # Each fresh yellow observation publishes its own reconstructed path;
        # do not require agreement with an old FULL or previous PARTIAL path.
        self._recovery_consecutive+=1
        self._recovery_prev_world_path=np.array(world,dtype=float,copy=True)
        self._recovery_prev_world_time=now
        span=float(info.get('accepted_span_m',0.0))
        # A geometrically accepted path is NOT automatically a drive command.
        # Physically driving requires validated camera mounting, a corridor check,
        # fresh yellow observations, consecutive geometry agreement, and a limited
        # travel/time envelope anchored in a real FULL observation.
        lateral_error=float(state[0])
        heading_error=float(state[1])
        lane_half_width=0.5*float(p('lane_width_m'))
        lateral_limit=min(max(0.,lane_half_width-0.02),
                          max(0.,float(p('recovery_drive_max_abs_cte_m'))))
        heading_limit=min(.35,max(0.,float(p('recovery_drive_max_abs_heading_rad'))))
        drive_checks={
            'ground_geometry_verified':bool(p('recovery_ground_geometry_verified')),
            'within_full_time':age<=min(1.5,float(p('recovery_drive_max_age_sec'))),
            'within_full_travel':self._recovery_travel_m<=min(.45,float(p('recovery_drive_max_travel_m'))),
            'enough_arc_preview':span>=max(.30,float(p('recovery_drive_min_span_m'))),
            'full_observed_geometry':not bool(info.get('nearfield_only',False)),
            'enough_fresh_yellow_frames':self._recovery_consecutive>=max(4,int(p('recovery_drive_min_consecutive'))),
            'lane_corridor_plausible':abs(lateral_error)<=lateral_limit,
            'heading_plausible':abs(heading_error)<=heading_limit,
            # When the entire recovered centreline is outside the image the
            # road occupancy can't be verified with this monocular view.
            'centre_visible_to_camera':not bool(geometry.get('centre_off_camera',False)),
        }
        drive_eligible=all(drive_checks.values())
        path=ROSPath()
        path.header.stamp=self.get_clock().now().to_msg()
        path.header.frame_id=frame
        for xx,yy in world:
            pose=PoseStamped();pose.header=path.header
            pose.pose.position.x=float(xx);pose.pose.position.y=float(yy)
            pose.pose.orientation.w=1.0
            path.poses.append(pose)
        stamp_ns=int(path.header.stamp.sec)*1000000000+int(path.header.stamp.nanosec)
        evidence=String()
        evidence.data=json.dumps({
            'protocol':'qcar_yellow_ground_v6','valid':bool(drive_eligible),
            'geometry_valid':True,'fresh_yellow_observed':True,
            'source':'pinhole_ground_height_pitch','reconstruction':'curve_normal_v7_full_aligned',
            'full_alignment_samples':int(getattr(self,'_recovery_alignment_samples',0)),
            'verified_ground_geometry':bool(p('recovery_ground_geometry_verified')),
            'ground_camera_height_m':float(p('ground_camera_height_m')),
            'ground_camera_pitch_rad':float(p('ground_camera_pitch_rad')),
            'measured_lane_width_m':float(p('lane_width_m')),
            'recovery_cte_m':float(lateral_error),
            'recovery_heading_error_rad':float(heading_error),
            'drive_checks':drive_checks,
            'centre_off_camera':bool(geometry.get('centre_off_camera',False)),
            'fit_model':str(geometry['fit_model']),
            'fit_rmse_m':float(geometry['fit_rmse_m']),
            'inlier_fraction':float(geometry['inlier_fraction']),
            'normal_offset_m':.5*float(p('lane_width_m')),
            'path_stamp_ns':stamp_ns,
            'age_sec':float(age),'distance_since_full_m':float(self._recovery_travel_m),
            'consecutive_frames':int(self._recovery_consecutive),
            'points':len(path.poses),'nearfield_only':bool(info.get('nearfield_only',False)),
            'accepted_span_m':span,
            'p95_curvature_m_inv':float(info.get('p95_curvature_m_inv',0.)),
            'motion_window_eligible':bool(drive_eligible),'visual_only':not drive_eligible})
        self._recovery_evidence_pub.publish(evidence)
        self._publish_or_delay_lane_path(path)
        self._recovery_valid_pub.publish(Float64(data=1.0))
        self._publish_lane_quality(True,
            ('PARTIAL live-yellow metric path eligible for independently gated driving'
             if drive_eligible else 'PARTIAL live-yellow metric path; visualization only'),
            span_m=span,p95=info.get('p95_curvature_m_inv'))
        self._recovery_diagnostic(True,'accepted live yellow ground-scale estimate',
             **basic,continuous_frames=int(self._recovery_consecutive),
             ground_scale='fixed physical height and provisional pitch',travel_m=float(self._recovery_travel_m),
             odom_step_m=float(odom_step_m),
             odom_step_dt_sec=float(odom_step_dt_sec),
             motion_window_eligible=bool(drive_eligible),
             drive_checks=drive_checks,
             centre_off_camera=bool(geometry.get('centre_off_camera',False)),
             debug_visible_points=int(geometry.get('debug_visible_points',0)),
             fit_model=str(geometry['fit_model']),fit_rmse_m=float(geometry['fit_rmse_m']),
             inlier_fraction=float(geometry['inlier_fraction']),
             accepted_span_m=span,nearfield_only=bool(info.get('nearfield_only',False)),
             p95_curvature_m_inv=info.get('p95_curvature_m_inv'))
        self._current_metric_path=np.asarray(cleaned,float)
        return dbg,state

    # ==================================================================
    # FULL-observed alignment of yellow-only recovery (no synthetic lane pixels)
    # ==================================================================
    @staticmethod
    def _fractional_lane_samples(path, count=20):
        pts = np.asarray(path, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
            return None
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        good = np.r_[True, np.diff(cumulative) > 1.0e-5]
        cumulative, pts = cumulative[good], pts[good]
        if len(cumulative) < 8 or cumulative[-1] < .15:
            return None
        query = np.linspace(0., cumulative[-1], count)
        return np.column_stack((np.interp(query, cumulative, pts[:, 0]),
                                np.interp(query, cumulative, pts[:, 1])))

    def _learn_full_to_yellow_alignment(self, left_points):
        """Learn only from CURRENT frames with BOTH real lane boundaries.

        The current FULL metric path is the V23 two-boundary reference used by
        the controller; the yellow-only path is projected from the same image.
        Fit a best 2D similarity transform between equally travelled portions
        of these two measured curves. This prevents an origin/scale jump when
        only yellow survives; it does not create any new observed lane pixels.
        """
        p = lambda name: self.get_parameter(name).value
        full = getattr(self, '_current_metric_path', None)
        if full is None or not left_points:
            return
        if any(getattr(self, k) is None for k in
               ('_camera_fx', '_camera_fy', '_camera_cx', '_camera_cy')):
            return
        ypath, _, meta = recover_ground_curve_normal_from_yellow(
            left_points, fx=float(self._camera_fx), fy=float(self._camera_fy),
            cx=float(self._camera_cx), cy=float(self._camera_cy),
            camera_height_m=float(p('ground_camera_height_m')),
            camera_pitch_rad=float(p('ground_camera_pitch_rad')),
            lane_width_m=float(p('lane_width_m')),
            max_range_m=float(p('ground_max_recovery_range_m')),
            fit_max_rmse_m=float(p('recovery_curve_fit_max_rmse_m')),
            min_inlier_fraction=float(p('recovery_curve_min_inlier_fraction')))
        if ypath is None:
            return
        src = self._fractional_lane_samples(ypath)
        dst = self._fractional_lane_samples(full)
        if src is None or dst is None:
            return
        s0, d0 = src.mean(axis=0), dst.mean(axis=0)
        S, D = src - s0, dst - d0
        H = S.T @ D
        try:
            U, singular, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return
        correction = np.eye(2)
        correction[-1,-1] = np.linalg.det(U @ Vt)
        R = U @ correction @ Vt
        scale = float(np.sum(singular * [1., correction[-1,-1]]) / np.sum(S*S))
        if not np.isfinite(scale) or not (0.30 < scale < 2.00):
            return
        t = d0 - scale * s0 @ R
        residual = float(np.sqrt(np.mean(np.sum((scale * src @ R + t - dst)**2, axis=1))))
        if residual > .080:
            return  # Reject numerically inconsistent paired camera observations.
        # Avoid oscillating calibration coefficients between adjacent FULL frames.
        # This is a model calibration cache, NOT a new driving authorization gate.
        alpha = .30
        model = np.r_[scale, math.atan2(R[0,1], R[0,0]), t]
        prior = getattr(self, '_last_recovery_alignment', None)
        if prior is not None:
            model = (1.-alpha)*prior + alpha*model
        self._last_recovery_alignment = model
        self._recovery_alignment_samples = getattr(self, '_recovery_alignment_samples', 0)+1
        self._recovery_alignment_rmse = residual

    def _align_yellow_recovery(self, points):
        """Use the last actually measured FULL-to-yellow transform."""
        model = getattr(self, '_last_recovery_alignment', None)
        if model is None:
            return points
        scale, theta, tx, ty = model
        cs, sn = math.cos(theta), math.sin(theta)
        R = np.array([[cs, sn],[-sn, cs]],dtype=float)
        return scale*np.asarray(points,float)@R + np.array([tx,ty])

    # ==================================================================
    # Publishing
    # ==================================================================
    def _publish_lane_quality(self, accepted, reason, *, span_m=None, p95=None):
        """Machine-readable perception diagnostics; no invented lane geometry."""
        if not hasattr(self, '_lane_path_quality_pub'):
            return
        result = {'accepted': bool(accepted), 'reason': str(reason)}
        if span_m is not None and math.isfinite(float(span_m)):
            result['observed_forward_span_m'] = float(span_m)
        if p95 is not None and math.isfinite(float(p95)):
            result['p95_curvature_m_inv'] = float(p95)
        msg = String()
        msg.data = json.dumps(result, sort_keys=True)
        self._lane_path_quality_pub.publish(msg)

    def _warn_path_blocked(self, reason):
        """Publish every rejection reason; rate-limit only the terminal warning."""
        self._publish_lane_quality(False, reason)
        now = time.monotonic()
        if now - getattr(self, '_last_path_skip_log_time', float('-inf')) < 3.0:
            return
        self._last_path_skip_log_time = now
        if hasattr(self, 'get_logger'):
            self.get_logger().warning('Lane reference path withheld: ' + str(reason))

    def _publish_or_delay_lane_path(self, path):
        """Publish immediately normally; otherwise queue unchanged ROS Path."""
        if self._path_delay_buffer is None:
            self._lane_path_pub.publish(path)
            return
        with self._path_delay_cv:
            if self._path_delay_shutdown:
                return
            self._path_delay_buffer.push(path, time.monotonic())
            self._path_delay_cv.notify()

    def _release_delayed_paths(self):
        """Release queued Paths at steady-clock deadlines, independently of CV."""
        while True:
            with self._path_delay_cv:
                ready = []
                while not self._path_delay_shutdown:
                    now = time.monotonic()
                    ready = self._path_delay_buffer.pop_ready(now)
                    if ready:
                        break
                    deadline = self._path_delay_buffer.next_release_time()
                    timeout = None if deadline is None else max(0.0, deadline - now)
                    self._path_delay_cv.wait(timeout=timeout)
                if self._path_delay_shutdown:
                    return
            # Publishing outside the lock lets the camera thread enqueue freely.
            for path, actual_ms in ready:
                if self._path_delay_shutdown:
                    return
                try:
                    self._lane_path_pub.publish(path)
                    actual_msg = Float64()
                    actual_msg.data = float(actual_ms)
                    self._actual_delay_pub.publish(actual_msg)
                except Exception as exc:
                    if not self._path_delay_shutdown:
                        self.get_logger().error(
                            'Delayed lane path publish failed: {}'.format(exc)
                        )
                    return

    def destroy_node(self):
        """Ensure delayed messages are not published after node shutdown."""
        if self._path_delay_cv is not None:
            with self._path_delay_cv:
                self._path_delay_shutdown = True
                self._path_delay_buffer.clear()
                self._path_delay_cv.notify_all()
            if self._path_delay_thread is not None:
                self._path_delay_thread.join(timeout=2.0)
        return super().destroy_node()

    def _publish_reference_path(self, metric_path, status):
        """Publish the observed V23 centerline for rcognita in the odom frame.

        This adds no extrapolation or independent prediction beyond V23 itself.
        V23 may briefly hold a yellow trace (yellow_hold_last_sec, default 0.18).
        If lane geometry or odometry is unavailable, publish NOTHING and let
        the ROS controller's lane-mode path timeout stop the vehicle.
        """
        p = lambda name: self.get_parameter(name).value
        if not bool(p('publish_lane_path')):
            return
        if int(status) != int(LANE_STATUS_FULL):
            self._warn_path_blocked('cannot publish lane path without BOTH measured boundaries')
            return
        if int(status) < int(p('min_lane_status_for_path')):
            self._warn_path_blocked('lane status below min_lane_status_for_path')
            return
        odom_pose, odom_last, _ = self._odom_snapshot()
        if odom_pose is None or odom_last is None:
            self._warn_path_blocked('no /odom/bot received')
            return
        age = time.monotonic() - odom_last
        if age < 0.0 or age > float(p('odom_freshness_sec')):
            self._warn_path_blocked(f'odometry stale: {age:.2f}s')
            return
        if bool(p('require_camera_info_for_path')) and self._camera_fx is None:
            self._warn_path_blocked(
                'camera calibration missing; check /front_camera/camera_info')
            return
        if metric_path is None:
            self._warn_path_blocked('no metric centerline samples')
            return
        points = np.asarray(metric_path, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            return
        # No synthetic extrapolation: reject if the observed point set is too
        # short or degenerate. Preserve near-to-far sample order.
        if len(points) < max(3, int(p('min_path_points'))):
            self._warn_path_blocked('too few centerline samples')
            return
        if not np.all(np.isfinite(points)):
            return
        finite_increments = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5
        points = points[np.r_[True, finite_increments]]
        if len(points) < max(3, int(p('min_path_points'))):
            self._warn_path_blocked('too few centerline samples')
            return
        if (float(points[-1, 0] - points[0, 0])
                < float(p('min_path_forward_span_m'))):
            self._warn_path_blocked('centerline has insufficient forward span')
            return
        # Smooth/resample measured metric centreline *before* rotating to odom.
        # In particular do not send 2-mm depth-derived segments to MPC, where
        # derivative-of-heading would report artificial high curvature.
        cleaned, quality = sanitize_metric_lane_path(
            points,
            spacing_m=float(p('lane_path_spacing_m')),
            fit_radius_m=float(p('lane_path_fit_radius_m')),
            min_span_m=max(float(p('lane_path_min_span_m')), float(p('min_path_forward_span_m'))),
            max_curvature_m_inv=float(p('lane_path_max_curvature_m_inv')),
            max_observation_gap_m=float(p('lane_path_max_observation_gap_m')),
        )
        if cleaned is None:
            self._warn_path_blocked('metric lane geometry rejected: {} (span={:.2f} m, '
                                    'p95 curvature={:.2f} 1/m)'.format(
                quality['reason'], quality['span_m'], quality['p95_curvature_m_inv']))
            return
        points = cleaned
        base_x, base_y, base_yaw, frame = odom_pose
        try:
            xy_world = camera_centerline_to_world(
                points[:, 0], points[:, 1],
                base_x, base_y, base_yaw,
                float(p('camera_x_from_base_m')),
                float(p('camera_y_from_base_m')),
                float(p('camera_yaw_from_base_rad')),
            )
        except ValueError:
            self._warn_path_blocked('camera-to-odom coordinate conversion failed')
            return
        path = ROSPath()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = frame
        for px, py in xy_world:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(px)
            pose.pose.position.y = float(py)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._publish_or_delay_lane_path(path)
        # Save only published, validated FULL geometry. A failed FULL frame
        # never updates the recovery history.
        # Keep latest FULL path separate until row-ground-scale learning succeeds.
        # Otherwise an uncalibratable FULL frame would invalidate the paired
        # previous scale/path and make the next PARTIAL recovery inconsistent.
        self._latest_accepted_full_world_path = np.asarray(xy_world, dtype=float).copy()
        self._publish_lane_quality(True, 'accepted measured two-boundary path',
                                   span_m=quality.get('span_m'),
                                   p95=quality.get('p95_curvature_m_inv'))
        return True

    def _publish_status(self, status):
        msg = Float64()
        msg.data = float(status)
        self._status_pub.publish(msg)

    def _publish_state(
        self,
        cte,
        heading,
        curvature,
        status,
    ):
        msg = Float64()

        msg.data = float(cte)
        self._cte_pub.publish(msg)

        msg = Float64()
        msg.data = float(heading)
        self._heading_pub.publish(msg)

        msg = Float64()
        msg.data = float(curvature)
        self._curvature_pub.publish(msg)

        self._publish_status(status)

    def _publish_mono(self, publisher, roi_mask, frame_shape, roi_top, frame_id):
        full = np.zeros(frame_shape[:2], dtype=np.uint8)
        full[roi_top:] = roi_mask

        msg = self._bridge.cv2_to_imgmsg(
            full,
            encoding='mono8',
        )
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        publisher.publish(msg)

    def _publish_debug(
        self,
        frame,
        roi_top,
        yellow_mask,
        right_edge_mask,
        road_mask,
        corridor,
        left_points,
        right_points,
        centre_points,
        centre_source,
        raw_state,
        status,
        frame_id,
    ):
        debug = frame.copy()
        img_h, img_w = debug.shape[:2]

        cv2.line(
            debug,
            (0, roi_top),
            (img_w - 1, roi_top),
            (255, 255, 0),
            1,
        )

        # Friend-style: draw only actual sampled/smoothed points, not a global
        # polynomial that can curl into unsupported parts of the image.
        self._draw_points(
            debug,
            left_points,
            (0, 255, 0),
            thickness=3,
        )
        self._draw_points(
            debug,
            right_points,
            (0, 0, 255),
            thickness=3,
        )

        centre_xy = [
            (item[0], item[1])
            for item in centre_points
        ]
        self._draw_points(
            debug,
            centre_xy,
            (255, 0, 255),
            thickness=2,
        )

        if raw_state is not None:
            cte = (
                self._filtered_cte
                if self._filtered_cte is not None
                else raw_state[0]
            )
            heading = (
                self._filtered_heading
                if self._filtered_heading is not None
                else raw_state[1]
            )
            curvature = (
                self._filtered_curvature
                if self._filtered_curvature is not None
                else raw_state[2]
            )

            if centre_points:
                x0 = int(round(centre_points[0][0]))
                y0 = int(round(centre_points[0][1]))

                length = 55
                dx = int(
                    round(
                        length
                        * math.sin(float(heading))
                    )
                )
                dy = int(
                    round(
                        length
                        * math.cos(float(heading))
                    )
                )

                cv2.line(
                    debug,
                    (x0, y0),
                    (x0 + dx, y0 - dy),
                    (0, 255, 255),
                    3,
                )
        else:
            cte = None
            heading = None
            curvature = None

        # Vehicle reference.
        cv2.circle(
            debug,
            (img_w // 2, img_h - 8),
            5,
            (0, 255, 255),
            -1,
        )

        y_text = 24

        def put(text, colour=(255, 255, 255)):
            nonlocal y_text
            cv2.putText(
                debug,
                text,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                1,
                cv2.LINE_AA,
            )
            y_text += 20

        if raw_state is not None:
            put(
                'CTE:        {:+.4f} m'.format(cte),
                (0, 220, 255),
            )
            put(
                'Heading:    {:+.2f} deg'.format(
                    math.degrees(heading)
                ),
                (255, 0, 255),
            )
            put(
                'Curvature:  {:+.4f} 1/m'.format(curvature),
                (0, 255, 180),
            )
        else:
            put('CTE/heading/curvature: INVALID', (0, 0, 255))

        put('Speed:      {:.3f} m/s'.format(self._speed))
        put('Centre:     {}'.format(centre_source))
        put('Right:      {} | forward {:.2f} m'.format(
            self._right_source, self._right_span_m))

        status_name = {
            LANE_STATUS_FULL: 'FULL / yellow + right edge',
            LANE_STATUS_PARTIAL: 'PARTIAL / one boundary',
            LANE_STATUS_LOST: 'LOST',
        }.get(float(status), str(status))

        put('Lane status: {}'.format(status_name))

        # V22 top-right inset:
        # Show the ACTUAL boundary evidence being used on this frame.
        #
        # - yellow boundary pixels, if yellow is detected
        # - right/white edge pixels, if the right boundary is detected
        #
        # This is intentionally different from the old corridor-only inset.
        # If only the white/right lane is visible, the inset must still show it.
        inset_mask = np.zeros_like(corridor, dtype=np.uint8)

        # Show only the tracked yellow path, not all raw yellow threshold pixels.
        for x, y_full in left_points:
            y = int(round(float(y_full) - float(roi_top)))
            xi = int(round(float(x)))
            if (
                0 <= y < inset_mask.shape[0]
                and 0 <= xi < inset_mask.shape[1]
            ):
                inset_mask[
                    y,
                    max(0, xi - 2):min(inset_mask.shape[1], xi + 3),
                ] = 255

        # Right-edge mask already contains only the accepted physical
        # road/white boundary points.
        inset_mask = cv2.bitwise_or(
            inset_mask,
            right_edge_mask,
        )

        thumb_w = min(160, img_w)
        thumb_h = min(90, img_h)

        inset = cv2.resize(
            cv2.cvtColor(inset_mask, cv2.COLOR_GRAY2BGR),
            (thumb_w, thumb_h),
            interpolation=cv2.INTER_NEAREST,
        )

        x0 = img_w - thumb_w - 5
        if x0 >= 0 and 5 + thumb_h <= img_h:
            debug[
                5:5 + thumb_h,
                x0:x0 + thumb_w,
            ] = inset

        # --------------------------------------------------------------
        # V23: publish BOTH raw and compressed debug streams.
        #
        # The compressed stream is intended for continuous rqt viewing.
        # The raw stream is retained for exact pixel inspection, but defaults
        # to every 3rd frame so it cannot monopolize bandwidth/serialization.
        # --------------------------------------------------------------
        self._debug_frame_counter += 1
        debug_stamp = self.get_clock().now().to_msg()

        if bool(self.get_parameter('publish_raw_debug').value):
            raw_divisor = max(
                1,
                int(
                    self.get_parameter(
                        'raw_debug_publish_divisor'
                    ).value
                ),
            )

            if self._debug_frame_counter % raw_divisor == 0:
                msg = self._bridge.cv2_to_imgmsg(
                    debug,
                    encoding='bgr8',
                )
                msg.header.stamp = debug_stamp
                msg.header.frame_id = frame_id
                self._debug_pub.publish(msg)

        if bool(self.get_parameter('publish_compressed_debug').value):
            compressed_frame = debug

            target_width = int(
                self.get_parameter(
                    'compressed_debug_width'
                ).value
            )

            if (
                target_width > 0
                and compressed_frame.shape[1] > target_width
            ):
                scale = (
                    float(target_width)
                    / float(compressed_frame.shape[1])
                )
                target_height = max(
                    1,
                    int(round(compressed_frame.shape[0] * scale)),
                )
                compressed_frame = cv2.resize(
                    compressed_frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )

            jpeg_quality = int(
                clamp(
                    float(
                        self.get_parameter(
                            'debug_jpeg_quality'
                        ).value
                    ),
                    20.0,
                    95.0,
                )
            )

            ok, encoded = cv2.imencode(
                '.jpg',
                compressed_frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    jpeg_quality,
                ],
            )

            if ok:
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = debug_stamp
                compressed_msg.header.frame_id = frame_id
                compressed_msg.format = 'jpeg'
                compressed_msg.data = encoded.tobytes()
                self._debug_compressed_pub.publish(
                    compressed_msg
                )

        self._publish_mono(
            self._yellow_mask_pub,
            yellow_mask,
            frame.shape,
            roi_top,
            frame_id,
        )
        self._publish_mono(
            self._right_edge_pub,
            right_edge_mask,
            frame.shape,
            roi_top,
            frame_id,
        )
        if road_mask is None:
            road_debug = np.zeros_like(yellow_mask)
        else:
            road_debug = road_mask
        self._publish_mono(
            self._road_mask_pub,
            road_debug,
            frame.shape,
            roi_top,
            frame_id,
        )
        self._publish_mono(
            self._corridor_pub,
            corridor,
            frame.shape,
            roi_top,
            frame_id,
        )

    @staticmethod
    def _draw_points(image, points, colour, thickness=2):
        if points is None or len(points) < 2:
            return

        pts = np.asarray(
            [
                [int(round(item[0])), int(round(item[1]))]
                for item in points
            ],
            dtype=np.int32,
        )

        cv2.polylines(
            image,
            [pts],
            False,
            colour,
            thickness,
            cv2.LINE_AA,
        )

        # Small sampled-point markers reproduce the clean reference-style
        # visualization and make it obvious which rows are actually observed.
        for index in range(0, len(pts), 3):
            cv2.circle(
                image,
                tuple(pts[index]),
                2,
                colour,
                -1,
            )


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

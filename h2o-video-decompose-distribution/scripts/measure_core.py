"""Registration and edge-profile estimators shared with the source analysis."""

import cv2 as cv
import numpy as np
from scipy.special import erf


def project(h, points):
    xyz = np.column_stack([points, np.ones(len(points))]) @ np.asarray(h).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return xyz[:, :2] / xyz[:, 2:]


def normalize(img):
    raw = cv.GaussianBlur(img.astype("float32"), (5, 5), .9)
    return raw - cv.GaussianBlur(raw, (0, 0), 6)


def lattice(p,w=34,h=31):
    return np.floor(p/np.array([w,h])).astype(int)
def holdout(p):
    cell=lattice(p)
    return (cell[:,0]*23+cell[:,1]*37)%5==2

def solve_pair(a,b,strict=True):
    kind=[]; errors={}
    if len(a)<5: return None
    hold=holdout(a)
    if hold.sum()<3 or (~hold).sum()<6:
        if strict: return None
        hold=np.zeros(len(a),bool)
    u=np.ascontiguousarray(a[~hold]); v=np.ascontiguousarray(b[~hold])
    for key,api,nmin in [('similarity',cv.estimateAffinePartial2D,4),('affine',cv.estimateAffine2D,5),
                        ('homography',None,10)]:
        if len(u)<nmin: continue
        if api:
            m,ins=api(u,v,method=cv.RANSAC,ransacReprojThreshold=2.8,refineIters=12)
            if m is not None: m=np.vstack([m,[0,0,1]])
        else:
            m,ins=cv.findHomography(u,v,cv.USAC_MAGSAC,2.8)
        if m is None or np.sum(ins)<nmin or not np.all(np.isfinite(m)): continue
        if np.linalg.det(m)<=0: continue
        # Check perspective only within the observed area. Reject folding and near poles.
        lo=np.min(u,axis=0); hi=np.max(u,axis=0)
        d=np.array([[lo[0],lo[1],1],[hi[0],lo[1],1],[hi[0],hi[1],1],[lo[0],hi[1],1]])@m[2]
        if np.min(d)<=.15 or np.max(d)/np.min(d)>4: continue
        kk=hold if np.any(hold) else ~hold
        err=np.linalg.norm(project(m,a[kk])-b[kk],axis=1)
        p50,p95=np.quantile(err,[.5,.95])
        n=int(np.sum(ins)); errors[key]=[float(p50),float(p95)]
        # Prefer the simpler model when its predictive error is practically equivalent.
        score=float(p50+.30*p95+ {'similarity':0,'affine':.13,'homography':.29}[key])
        kind.append((score,key,m,n,p50,p95))
    if not kind: return None
    _,key,m,n,p50,p95=min(kind,key=lambda x:x[0])
    return dict(model=key,H=(m/m[2,2]).tolist(),inlier_observations=n,
                holdout_count=int(np.sum(hold)),holdout_median_px=float(p50) if hold.any() else None,
                holdout_p95_px=float(p95) if hold.any() else None,model_errors_px=errors)

def pick(gray,mask,spacing=16):
    pt=cv.goodFeaturesToTrack(gray,240,.009,spacing,mask=mask,blockSize=5)
    return pt.reshape(-1,2) if pt is not None else np.empty((0,2),np.float32)
def flow(im,to,a,shift=None):
    if len(a)<4: return a[:0],a[:0]
    start=a.reshape(-1,1,2).astype('float32').copy()
    guess=start+np.float32(shift).reshape(1,1,2) if shift is not None else None
    opt=dict(winSize=(29,29),maxLevel=3,criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_COUNT,42,.015))
    target,ok,err=cv.calcOpticalFlowPyrLK(im,to,start,guess,flags=cv.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0,**opt)
    restored,bk,_=cv.calcOpticalFlowPyrLK(to,im,target,start.copy(),flags=cv.OPTFLOW_USE_INITIAL_FLOW,**opt)
    p=target[:,0,:]
    use=(ok.ravel()>0)&(bk.ravel()>0)&(np.linalg.norm(restored[:,0,:]-a,axis=1)<1.15)&(err.ravel()<26)&(
        (p[:,0]>3)&(p[:,0]<im.shape[1]-3)&(p[:,1]>4)&(p[:,1]<im.shape[0]-4))
    return a[use],p[use]

def ncc_matches(ref,image,points,h):
    # Only use the initial transform to narrow the search for the same patch.
    try: corrected=cv.warpPerspective(image,np.linalg.inv(h),(ref.shape[1],ref.shape[0]),flags=cv.INTER_LINEAR)
    except np.linalg.LinAlgError: return None
    frame=normalize(corrected)
    a=[]; b=[]
    for x,y in np.rint(points).astype(int):
        half=12; rad=16; total=half+rad
        if x-total<0 or x+total>=ref.shape[1] or y-total<0 or y+total>=ref.shape[0]: continue
        template=ref[y-half:y+half+1,x-half:x+half+1]
        search=frame[y-total:y+total+1,x-total:x+total+1]
        if np.std(template)<.85: continue
        corr=cv.matchTemplate(search,template,cv.TM_CCOEFF_NORMED)
        _,score,_,(cx,cy)=cv.minMaxLoc(corr)
        if score<.64 or not (0<cx<2*rad and 0<cy<2*rad): continue
        competing=corr.copy()
        competing[max(0,cy-3):cy+4,max(0,cx-3):cx+4]=-1
        if score-float(competing.max())<.045: continue
        def off(v):
            l,m,r=v
            return np.clip(.5*(l-r)/(l-2*m+r-1e-7),-.5,.5)
        a.append([x,y]); b.append([x+cx-rad+off(corr[cy,cx-1:cx+2]),y+cy-rad+off(corr[cy-1:cy+2,cx])])
    return np.array(a),project(h,np.array(b)) if b else np.empty((0,2))

SAMPLES=np.arange(-14,14.01,.5,dtype=np.float32)
SHIFTS=np.arange(-2,2.01,.5,dtype=np.float32)
SIGMAS=np.geomspace(.45,6.0,32).astype('float32')
PARAMS=np.array([(sigma,shift) for sigma in SIGMAS for shift in SHIFTS])
ESF=.5*(1+erf((SAMPLES[None,:]-PARAMS[:,1,None])/(2**.5*PARAMS[:,0,None])))
BASE=np.c_[np.ones(len(SAMPLES)),SAMPLES]
PROJ=np.eye(len(SAMPLES))-BASE@np.linalg.pinv(BASE)
TEMPLATES=ESF@PROJ
T_NORM=np.sum(TEMPLATES**2,axis=1)

def sample(image, dx, dy, mask, minimum_contrast=10):
    magnitude=np.hypot(dx,dy)
    # One candidate per cell, then gate by monotonicity and fit residual.
    ys,xs=np.where((mask!=0)&(magnitude>=1.35))
    ids=(ys//11)*((image.shape[1]+10)//11)+(xs//11)
    if not len(xs): return np.empty((0,6))
    order=np.argsort(-magnitude[ys,xs])
    _,first=np.unique(ids[order],return_index=True)
    y=ys[order[first]]; x=xs[order[first]]
    keep=(x>=15)&(x<image.shape[1]-15)&(y>=15)&(y<image.shape[0]-15)
    x=x[keep]; y=y[keep]
    if not len(x): return np.empty((0,6))
    gx=dx[y,x]/magnitude[y,x]
    gy=dy[y,x]/magnitude[y,x]
    profiles=[]
    for tangent in [-1.,0.,1.]:
        xx=x[:,None]+gx[:,None]*SAMPLES-gy[:,None]*tangent
        yy=y[:,None]+gy[:,None]*SAMPLES+gx[:,None]*tangent
        profiles.append(cv.remap(image,xx.astype('float32'),yy.astype('float32'),cv.INTER_LINEAR))
    v=np.mean(profiles,axis=0)
    early=np.median(v[:,:10],axis=1)
    late=np.median(v[:,-10:],axis=1)
    contrast=late-early
    detrend=v@PROJ
    strength=detrend@TEMPLATES.T
    coefficient=strength/T_NORM
    score=np.maximum(0,strength)**2/T_NORM
    best=np.argmax(score,axis=1)
    idx=np.arange(len(x)); a=coefficient[idx,best]
    err=np.sqrt(np.maximum(0,(np.sum(detrend**2,axis=1)-score[idx,best])/len(SAMPLES)))
    model=ESF[best]*a[:,None]
    offset_slope=(v-model)@np.linalg.pinv(BASE).T
    width=2.563103*PARAMS[best,0]
    # Keep separable rising edges. A thin stroke with a visible falling side
    # or a large local luminance gradient is rejected.
    valid=(a>=minimum_contrast)&(contrast>=.52*a)&(contrast<=1.48*a)&(err<=.11*a)&(
           np.abs(offset_slope[:,1])*28<=.40*a)&(width<15.1)
    return np.c_[x,y,width,np.abs(gx), a, err][valid]

def stats(a):
    if len(a)<3: return None
    order=np.argsort(a[:,2])
    sharp=a[order[:max(3,int(np.ceil(.35*len(a))))]]
    xnorm=a[:,3]>.85
    ynorm=a[:,3]<.52
    return dict(n=len(a),q30=float(np.percentile(a[:,2],30)),median=float(np.median(a[:,2])),
                sharp_xy=np.median(sharp[:,:2],axis=0).tolist(),
                xnormal=float(np.median(a[xnorm,2])) if sum(xnorm)>=4 else None,
                ynormal=float(np.median(a[ynorm,2])) if sum(ynorm)>=4 else None)


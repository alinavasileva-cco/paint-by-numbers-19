"""Procedural studio-lit 3D assets used in the ISSAM Glass Editorial presentation.
Requires numpy and Pillow. No source-deck data or Google Drive credentials.
Run: python render_glass.py --size 300
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
parser=argparse.ArgumentParser()
parser.add_argument("--size",type=int,default=300)
parser.add_argument("--out",default="assets")
args=parser.parse_args()
N=args.size; OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
X,Y=np.meshgrid(np.linspace(-1.9,1.9,N,dtype=np.float32),np.linspace(1.9,-1.9,N,dtype=np.float32))
O=np.stack([X,Y,np.full_like(X,5)],axis=-1)
D=np.array([0,0,-1],dtype=np.float32)
def norm(v): return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-7)
def sphere(p,c,r): return np.linalg.norm(p-np.asarray(c,dtype=np.float32),axis=-1)-r
def torus(p,c,R,r,angle=.6):
    q=p-np.asarray(c,dtype=np.float32); a=np.cos(angle); b=np.sin(angle)
    u=q[...,0]; v=a*q[...,1]-b*q[...,2]; z=b*q[...,1]+a*q[...,2]
    return np.sqrt((np.sqrt(u*u+v*v)-R)**2+z*z)-r
def capsule(p,a,b,r):
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32); v=b-a; q=p-a
    h=np.clip(np.sum(q*v,axis=-1)/np.sum(v*v),0,1)
    return np.linalg.norm(q-h[...,None]*v,axis=-1)-r
def env(v):
    v=norm(v); top=np.clip(v[...,1]*.5+.5,0,1)
    base=np.array([.15,.23,.22])+top[...,None]*np.array([.57,.59,.56])
    for direction,power,color in [((-.6,.7,.5),24,(1.5,1.5,1.5)),((.8,.2,.5),15,(.8,.85,1.05)),((-.2,-.85,.5),9,(.45,.55,.48))]:
        k=np.clip(np.sum(v*norm(np.array(direction)),axis=-1),0,1)**power
        base+=k[...,None]*np.array(color)
    return np.clip(base,0,1)
def render(name,parts,palette):
    def sdf(p):
        ds=np.stack([f(p) for f in parts],axis=-1)
        return ds.min(axis=-1),ds.argmin(axis=-1)
    t=np.zeros((N,N),np.float32)
    for _ in range(60):
        p=O+t[...,None]*D; d,_=sdf(p); active=(d>.002)&(t<9)
        t+=np.where(active,np.maximum(d,.001),0)
        if not active.any():break
    p=O+t[...,None]*D; dist,ids=sdf(p); mask=(dist<.007)&(t<9)
    normals=[]
    for a in range(3):
        e=np.zeros(3,dtype=np.float32); e[a]=.004
        normals.append((sdf(p+e)[0]-sdf(p-e)[0])/.008)
    n=norm(np.stack(normals,axis=-1)); cos=np.clip(np.sum(n*(-D),axis=-1),0,1)
    ref=D-2*np.sum(D*n,axis=-1)[...,None]*n
    fr=.055+.945*(1-cos)**4
    reflected=env(ref); transmitted=env(norm(D+.75*n)); tint=np.asarray(palette)[ids]
    light=np.clip(np.sum(n*norm(np.array([-.6,.8,1])),axis=-1),0,1)
    color=(.24+.40*light[...,None])*tint+.38*transmitted*tint+fr[...,None]*(.94*reflected+.12)
    spec=np.clip(np.sum(ref*norm(np.array([-.6,.7,.5])),axis=-1),0,1)**45
    color=np.clip(color+.65*spec[...,None],0,1)**.85
    alpha=np.where(mask,np.clip(.84+.15*fr,0,1),0)
    out=np.dstack([color,alpha]); out[~mask,:]=0
    Image.fromarray((out*255).astype("uint8"),"RGBA").save(OUT/(name+".png"))
sage=(.58,.78,.68); plum=(.58,.47,.78); gold=(.82,.72,.50); ice=(.80,.90,.91)
render("torus",[lambda p:torus(p,(-.2,.08,0),.95,.28,.72),lambda p:torus(p,(.35,-.12,-.2),.65,.20,-.60),lambda p:sphere(p,(.98,.73,.28),.25)],[sage,plum,gold])
centers=[(-.57,.49,.2),(.48,.62,-.1),(.60,-.46,.1),(-.54,-.52,-.15)]
render("molecule",[lambda p,c=c,r=r:sphere(p,c,r) for c,r in zip(centers,[.44,.43,.46,.39])]+[lambda p:sphere(p,(.03,.04,.38),.43)]+[lambda p,c=c:capsule(p,c,(.03,.04,.38),.12) for c in centers],[sage,plum,gold,ice,sage,ice,ice,ice,ice])
render("breath",[lambda p:torus(p,(-.53,.05,0),.51,.20,.85),lambda p:torus(p,(.53,.05,.1),.51,.20,-.85),lambda p:capsule(p,(0,.1,.15),(0,1.05,.15),.17),lambda p:capsule(p,(0,.1,.15),(-.48,.3,0),.13),lambda p:capsule(p,(0,.1,.15),(.48,.3,0),.13)],[sage,plum,ice,ice,ice])
render("cold",[lambda p:sphere(p,(0,0,0),.42)]+[lambda p,a=a:capsule(p,(0,0,0),(np.cos(a)*1.2,np.sin(a)*1.2,.12),.12) for a in np.arange(6)*np.pi/3]+[lambda p,a=a:sphere(p,(np.cos(a)*1.2,np.sin(a)*1.2,.12),.20) for a in np.arange(6)*np.pi/3],[ice]+[sage]*6+[ice]*6)

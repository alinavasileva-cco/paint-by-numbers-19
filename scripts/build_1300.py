#!/usr/bin/env python3
"""Paint-by-numbers: geometry-first cleanup. No artificial circular label pockets.
Original source is NEVER edited. Every remaining color island gets a 5.3pt label.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import cairosvg
import build as previous

TARGET = 1300
FONT_PT = 5.3
STROKE_MM = .2
PAGE = (400., 600.)


def contacts(ids):
    """Shared edge length with 8-connected fallback for diagonal islands."""
    table = defaultdict(lambda: defaultdict(float))
    for a,b,weight in (
        (ids[:,:-1],ids[:,1:],1.),
        (ids[:-1,:],ids[1:,:],1.),
        (ids[:-1,:-1],ids[1:,1:],.12),
        (ids[1:,:-1],ids[:-1,1:],.12),
    ):
        mask = a != b
        v = np.stack((np.minimum(a[mask],b[mask]),
                      np.maximum(a[mask],b[mask])),axis=1)
        unique, lengths = np.unique(v,axis=0,return_counts=True)
        for (x,y),ln in zip(unique,lengths):
            if x < 1: continue
            table[int(x)][int(y)] += float(ln)*weight
            table[int(y)][int(x)] += float(ln)*weight
    return table


def fit_centers(ids, colors, palette, scale):
    """Find an actually rectangular number clearance for every island.

    At 400mm print width, 5.3pt ~= 5.25 source pixels high.
    Reserve 7 pixels height and 7/11 width incl. margin.
    """
    n=int(ids.max())
    ordering=previous.palette_indices(palette)
    codes={color:k+1 for k,color in enumerate(ordering)}
    numbers=np.zeros(n+1,dtype=np.int16)
    for r in range(1,n+1):
        numbers[r]=codes[int(colors[r])]
    raster=ids.astype(np.uint16)
    kernels=(np.ones((7,7),np.uint8),np.ones((7,11),np.uint8))
    good=[]
    for kernel in kernels:
        mn=cv2.erode(raster,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        mx=cv2.dilate(raster,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        good.append((mn==mx) & (raster!=0))
    wanted=np.where(numbers[raster]<=9,good[0],good[1])
    counts=np.bincount(ids[wanted],minlength=n+1)
    # Pick a safe center far away from any shared color boundary.
    border=np.zeros(ids.shape,dtype=bool)
    d=ids[:,:-1]!=ids[:,1:]
    border[:,:-1]|=d; border[:,1:]|=d
    d=ids[:-1,:]!=ids[1:,:]
    border[:-1,:]|=d; border[1:,:]|=d
    border[0,:]=True; border[-1,:]=True
    border[:,0]=True; border[:,-1]=True
    dist=cv2.distanceTransform((~border).astype(np.uint8),cv2.DIST_L2,5)
    coords=ndi.maximum_position(dist*wanted,labels=ids,index=np.arange(1,n+1))
    centers=np.zeros((n+1,2),np.int32)
    for r,xy in enumerate(coords,1):
        y,x=xy
        centers[r]=int(x),int(y)
    return counts[1:]>0,centers,numbers


def remap_colors(ids, colors, target):
    # Every pixel is reassigned to a color from the original 19-color palette.
    recolored=colors.copy()
    for r,c in target.items(): recolored[r]=c
    return recolored[ids]


def merge_until_fit(indexed, palette, scale, max_rounds=14):
    history=[]
    pigment_count=len(palette)
    rgb=np.asarray(palette,dtype=np.uint8)
    lab=cv2.cvtColor(rgb[None,:,:],cv2.COLOR_RGB2LAB)[0].astype(float)
    work=indexed.copy()
    for iteration in range(max_rounds):
        ids, colors, areas=previous.label_components(work,pigment_count)
        n=len(colors)-1
        valid,centers,numbers=fit_centers(ids,colors,palette,scale)
        bad=np.flatnonzero(~valid)+1
        history.append({'round':iteration,'regions':n,'fit':int(valid.sum()),'unfit':len(bad)})
        if len(bad)==0:
            return work,ids,colors,centers,numbers,history
        graph=contacts(ids)
        per_color=np.bincount(colors[1:].astype(np.int32),minlength=pigment_count)
        target={}
        # The smallest unpaintable flecks are merged first. Large valid shapes
        # and salient foreground silhouettes are not touched.
        for r in sorted(bad,key=lambda z:int(areas[z])):
            adj=graph.get(int(r),{})
            if not adj: continue
            original_color=int(colors[r])
            # 19 is a MAXIMUM, not a requirement to retain an unpaintable pigment.
            x,y=centers[r]
            def rank(z):
                c=int(colors[z])
                color_dist=float(np.linalg.norm(lab[original_color]-lab[c]))
                common=float(adj[z])
                # Prefer strongly similar pigments and existing broad forms.
                # Distant lights retain warm colors when an alternative exists.
                warm_guard=(y<470 and original_color!=c and
                            palette[original_color,0]>160 and
                            palette[c,0]<85)
                return (-100 if warm_guard else 0,
                        1 if c==original_color else 0,
                        -color_dist, common, float(areas[z]))
            # Merge toward a larger, already numberable neighbor. Never
            # let two invalid patches swap pigments back and forth.
            stable=[z for z in adj if valid[z-1]]
            larger=[z for z in adj if areas[z]>areas[r] or
                    (areas[z]==areas[r] and z<r)]
            pool=stable or larger or list(adj)
            best=max(pool,key=lambda z:(int(valid[z-1]),float(areas[z]),rank(z)))
            target[int(r)]=int(colors[best])
            per_color[original_color]-=1
        if not target: break
        new=remap_colors(ids,colors,target)
        if np.array_equal(new,work): break
        work=new
    ids,colors,areas=previous.label_components(work,pigment_count)
    valid,centers,numbers=fit_centers(ids,colors,palette,scale)
    history.append({'round':'final','regions':len(colors)-1,
                    'fit':int(valid.sum()),'unfit':int(np.count_nonzero(~valid))})
    return work,ids,colors,centers,numbers,history


def smooth_shared_paths(raw, scale):
    """Corner-round existing shared vectors once; no independent polygon borders."""
    import re
    out=[]
    for part in re.findall(r'M[^M]+',raw):
        pairs=re.findall(r'[ML](-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)',part)
        if len(pairs)<3:
            out.append(part); continue
        p=np.asarray([(float(x),float(y)) for x,y in pairs],float)
        closed=part.strip().endswith('Z')
        if closed and np.linalg.norm(p[0]-p[-1])<.002:p=p[:-1]
        def coord(v):return f'{v[0]:.3f},{v[1]:.3f}'
        if closed:
            n=len(p)
            if n<3:out.append(part);continue
            v=p[0]-p[-1]; v=v/(np.linalg.norm(v)+1e-9)
            q=p[1]-p[0]; q=q/(np.linalg.norm(q)+1e-9)
            radius=min(1.*scale,.24*np.linalg.norm(p[0]-p[-1]),
                       .24*np.linalg.norm(p[1]-p[0]))
            commands=['M'+coord(p[0]-v*radius)]
            indices=range(n)
        else:
            commands=['M'+coord(p[0])]
            indices=range(1,len(p)-1)
        for i in indices:
            a,b,c=p[(i-1)%len(p)],p[i],p[(i+1)%len(p)]
            left=b-a;right=c-b
            d1=np.linalg.norm(left);d2=np.linalg.norm(right)
            if min(d1,d2)<.00001:continue
            v=left/d1; q=right/d2
            dot=float(np.dot(v,q))
            if dot>.95 or dot<-.97:
                commands.append('L'+coord(b));continue
            radius=min(1.*scale,.24*d1,.24*d2)
            commands.append('L'+coord(b-v*radius)+'Q'+coord(b)+' '+coord(b+q*radius))
        if closed:commands.append('Z')
        else:commands.append('L'+coord(p[-1]))
        out.append(''.join(commands))
    return ' '.join(out)


def face_stroke(scale,xoff,yoff):
    # Non-enclosing 0.2mm printed eye edges; not a separate paint pigment.
    def p(x,y):return f'{xoff+x*scale:.3f},{yoff+y*scale:.3f}'
    return ('M'+p(636,385)+' Q'+p(645,372)+' '+p(653,382)
            +' M'+p(639,395)+' Q'+p(646,400)+' '+p(653,393))


def build(source, out):
    out.mkdir(exist_ok=True,parents=True)
    raw=source.read_bytes()
    image=np.asarray(Image.open(source).convert('RGB'))
    palette, inverse=np.unique(image.reshape(-1,3),return_inverse=True,axis=0)
    if len(palette)!=19:raise RuntimeError(f'Source has {len(palette)} colors, expected 19')
    baseline=np.asarray(Image.open(out/'color_segmentation_review.png').convert('RGB'))
    if baseline.shape!=image.shape:raise RuntimeError('Baseline dimensions differ')
    known=np.zeros(image.shape[:2],bool)
    indexed=np.zeros(image.shape[:2],np.uint8)
    for c,rgb in enumerate(palette):
        same=np.all(baseline==rgb,axis=2)
        known|=same; indexed[same]=c
    if not np.all(known):raise RuntimeError('Unrecognized baseline color pixels')
    height,width=indexed.shape
    scale=min(PAGE[0]/width,PAGE[1]/height)
    wmm,hmm=width*scale,height*scale
    xoff,yoff=(PAGE[0]-wmm)/2,(PAGE[1]-hmm)/2
    output,ids,colors,centers,numbers,history=merge_until_fit(indexed,palette,scale)
    n=len(colors)-1
    valid,centers,numbers=fit_centers(ids,colors,palette,scale)
    disconnected=[]
    # Check topological correspondence of region IDs; no unnumbered islands.
    for r in range(1,n+1):
        count,_=cv2.connectedComponents((ids==r).astype(np.uint8),8)
        if count!=2:disconnected.append(int(r))
    unfit=np.flatnonzero(~valid)+1
    misses=int(np.count_nonzero(ids==0))
    used=int(np.unique(colors[1:]).size)
    report={
      'source_sha256':hashlib.sha256(raw).hexdigest(),
      'rendered_page_mm':[PAGE[0],PAGE[1]],
      'artwork_size_mm':[round(wmm,3),round(hmm,3)],
      'source_dimensions_px':[width,height],
      'target_regions':TARGET, 'regions_total':n,
      'regions_numbered':int(valid.sum()),
      'regions_without_number':int(len(unfit)),
      'all_regions_numbered':bool(np.all(valid)),
      'label_fit_passed_regions':int(valid.sum()),
      'label_fit_failed_regions':int(len(unfit)),
      'pixels_without_color_assignment':misses,
      'source_colors':len(palette),'colors_in_derived_image':used,
      'font_size_pt':FONT_PT,'contour_width_mm':STROKE_MM,
      'source_not_overwritten':True,
      'disconnected_region_ids':disconnected[:50],
      'unlabelled_region_ids':[int(r) for r in unfit[:50]],
      'merge_history':history,
      'status':'PASS' if not len(unfit) and not misses and not disconnected
              and used<=19 and 1150<=n<=1450 else
              ('ALL_NUMBERED_BELOW_TARGET' if not len(unfit) and not misses and not disconnected
               and used<=19 else 'NEEDS_ARTISTIC_REVISION')
    }
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if len(unfit) or misses or disconnected or used>19:
        raise RuntimeError(f'Failed strict all-numbered gate: {json.dumps(report,ensure_ascii=False)}')
    stroke,npaths=previous.trace_smooth_shared_interfaces(output,scale,xoff,yoff)
    stroke=smooth_shared_paths(stroke,scale)
    font_mm=FONT_PT*25.4/72
    xml=[f'<svg xmlns="http://www.w3.org/2000/svg" width="400mm" height="600mm" viewBox="0 0 400 600">',
         '<rect width="400" height="600" fill="white"/>',
         f'<g fill="none" stroke="#171717" stroke-width="{STROKE_MM}" stroke-linecap="round" stroke-linejoin="round">',
         f'<rect x="{xoff:.3f}" y="{yoff:.3f}" width="{wmm:.3f}" height="{hmm:.3f}"/>',
         f'<path d="{stroke}"/>',
         f'<path d="{face_stroke(scale,xoff,yoff)}" stroke-width="0.15"/>',
         '</g>',
         f'<g fill="#101010" font-family="DejaVu Sans" font-size="{font_mm:.5f}" text-anchor="middle" dominant-baseline="central">']
    for r in range(1,n+1):
        x,y=centers[r]
        xml.append(f'<text x="{xoff+(x+.5)*scale:.3f}" y="{yoff+(y+.5)*scale:.3f}">{int(numbers[r])}</text>')
    xml.append('</g></svg>')
    (out/'lineart_review.svg').write_text('\n'.join(xml),encoding='utf-8')
    flat=palette[output]
    Image.fromarray(flat).save(out/'color_segmentation_review.png',optimize=True)
    Image.fromarray(flat).save(out/'color_artist_preview.png',optimize=True)
    Image.fromarray(ids.astype(np.uint16)).save(out/'region_ids_16bit.png')
    order=previous.palette_indices(palette)
    codes={c:k+1 for k,c in enumerate(order)}
    with (out/'palette.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Номер','HEX','R','G','B'])
        for c in order:
            R,G,B=map(int,palette[c])
            writer.writerow([codes[c],f'#{R:02X}{G:02X}{B:02X}',R,G,B])
    areas=np.bincount(ids.ravel(),minlength=n+1)
    with (out/'region_color_numbers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['region_id','color_number','HEX','area_px'])
        for r in range(1,n+1):
            c=int(colors[r]);R,G,B=map(int,palette[c])
            writer.writerow([r,int(numbers[r]),f'#{R:02X}{G:02X}{B:02X}',int(areas[r])])
    cairosvg.svg2png(url=str(out/'lineart_review.svg'),
                     write_to=str(out/'lineart_preview.png'),
                     output_width=2400,output_height=3600)
    cairosvg.svg2png(url=str(out/'lineart_review.svg'),
                     write_to=str(out/'lineart_screen.png'),
                     output_width=1100,output_height=1650)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.out)

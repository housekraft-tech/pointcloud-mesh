"""CPU depth-buffer previews of geometry read through the native SketchUp API.

These are geometry renders, explicitly not screenshots of the SketchUp UI.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def render(parts,path,title,offset=(1.3,-1.5,1.25),size=(1600,1150),edges=True,edge_depth_m=.015):
    width,height=size;canvas=np.full((height,width,3),[247,248,250],dtype=np.uint8)
    depth=np.full((height,width),-np.inf,dtype=np.float32)
    gradient_x=np.zeros((height,width),dtype=np.float32)
    gradient_y=np.zeros((height,width),dtype=np.float32)
    look=np.asarray(offset,dtype=float);look/=np.linalg.norm(look)
    up=np.array([0,0,1.]) if abs(look[2])<.99 else np.array([0,1.,0])
    right=np.cross(up,look);right/=np.linalg.norm(right);up=np.cross(look,right)
    vertices=np.vstack([p['v'] for p in parts]);basis=np.column_stack((right,up,look))
    projected=vertices@basis;lo,hi=projected[:,:2].min(0),projected[:,:2].max(0)
    scale=min((width-100)/(hi[0]-lo[0]),(height-160)/(hi[1]-lo[1]))
    centre=(hi+lo)/2;light=np.array([.3,-.4,.86]);light/=np.linalg.norm(light)
    total=0;started=time.time()
    for part in parts:
        v=np.asarray(part['v'],dtype=float);faces=np.asarray(part['f']);p=v@basis
        p[:,0]=(p[:,0]-centre[0])*scale+width/2
        p[:,1]=-(p[:,1]-centre[1])*scale+height/2+15
        triangle=p[faces];world=v[faces]
        normal=np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0])
        norm=np.linalg.norm(normal,axis=1);normal/=np.maximum(norm[:,None],1e-20)
        brightness=.65+.35*np.abs(normal@light)
        colours=np.clip(np.asarray(part.get('colour',[195,188,166]))[None,:]*brightness[:,None],0,255).astype(np.uint8)
        for t,colour in zip(triangle,colours):
            xmin=max(0,int(np.ceil(t[:,0].min())));xmax=min(width-1,int(np.floor(t[:,0].max())))
            ymin=max(75,int(np.ceil(t[:,1].min())));ymax=min(height-45,int(np.floor(t[:,1].max())))
            if xmin>xmax or ymin>ymax:continue
            a,b,c=t;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den)<1e-10:continue
            # Broadcast grids keep allocations bounded to the triangle extent.
            x=np.arange(xmin,xmax+1)[None,:];y=np.arange(ymin,ymax+1)[:,None]
            w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
            w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
            w2=1-w0-w1;inside=(w0>=-1e-8)&(w1>=-1e-8)&(w2>=-1e-8)
            z=w0*a[2]+w1*b[2]+w2*c[2]
            view=depth[ymin:ymax+1,xmin:xmax+1];keep=inside&(z>view)
            view[keep]=z[keep];canvas[ymin:ymax+1,xmin:xmax+1][keep]=colour
            gradient_x[ymin:ymax+1,xmin:xmax+1][keep]=((b[1]-c[1])*(a[2]-c[2])+(c[1]-a[1])*(b[2]-c[2]))/den
            gradient_y[ymin:ymax+1,xmin:xmax+1][keep]=((c[0]-b[0])*(a[2]-c[2])+(a[0]-c[0])*(b[2]-c[2]))/den
            total+=1
    if edges:
        # Dark lines where the depth buffer jumps: silhouettes, recesses and
        # steps between parallel faces that flat shading cannot separate.
        filled=np.isfinite(depth)
        jump=np.zeros_like(filled)
        for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
            a=depth[max(dy,0):height-max(-dy,0),max(dx,0):width-max(-dx,0)]
            b=depth[max(-dy,0):height-max(dy,0),max(-dx,0):width-max(dx,0)]
            both=np.isfinite(a)&np.isfinite(b)
            gx=gradient_x[max(dy,0):height-max(-dy,0),max(dx,0):width-max(-dx,0)]
            gy=gradient_y[max(dy,0):height-max(-dy,0),max(dx,0):width-max(-dx,0)]
            # Remove the depth change expected across the same flat face.
            # Without this, sloped walls/floors turn almost entirely black.
            diff=np.zeros(a.shape,bool)
            diff[both]=np.abs(a[both]-b[both]-gx[both]*dx-gy[both]*dy)>edge_depth_m
            jump[max(dy,0):height-max(-dy,0),max(dx,0):width-max(-dx,0)]|=diff&(a>=np.where(both,b,-np.inf))
        thick=jump.copy();thick[1:,:]|=jump[:-1,:];thick[:,1:]|=jump[:,:-1]
        canvas[thick&filled]=(canvas[thick&filled]*.25).astype(np.uint8)
    image=Image.fromarray(canvas);draw=ImageDraw.Draw(image)
    try:
        heading=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',27)
        footer=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    except OSError:heading=footer=ImageFont.load_default()
    draw.text((32,23),title,fill=(34,48,61),font=heading)
    draw.text((32,height-32),'Geometry read from the checked native .skp | CPU preview, not a SketchUp UI screenshot',fill=(73,80,88),font=footer)
    image.save(path)
    print(f'{Path(path).name}: {total:,} rasterized triangles, {time.time()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--label',required=True)
    args=parser.parse_args();folder=Path(args.folder)
    parts=json.loads((folder/'reopened_visible.build.json').read_text())['parts']
    render(parts,folder/'checked_house_3d.png',args.label+' | complete house')
    floor=[p for p in parts if 'floor' in p['kind'] or p['kind']=='ground_single_plane']
    if floor:render(floor,folder/'checked_floor_planes.png',args.label+' | floor planes and actual level differences')
    if len({p['level'] for p in parts})>1:
        for level in sorted({p['level'] for p in parts}):
            chosen=[p for p in parts if p['level']==level and 'ceiling' not in p['kind']]
            render(chosen,folder/f'checked_l{level}_3d.png',args.label+f' | level {level} interior cutaway')
        stairs=[p for p in parts if 'stair' in p['kind'] and p['kind'] not in ['stair_observed']]
        if stairs:render(stairs,folder/'checked_staircases.png',args.label+' | measured stair flights',offset=(1,-.3,.7))
        parking=[p for p in parts if 'plinth' in p['kind'] or p['name'] in ['Planar ground patch 29','Planar ground patch 30']]
        if parking:render(parking,folder/'checked_parking_connection.png',args.label+' | observed parking connection')

if __name__=='__main__':main()

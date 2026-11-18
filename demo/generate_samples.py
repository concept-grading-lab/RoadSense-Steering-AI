"""Generate original synthetic road scenes solely for demonstration, no real-driving data."""
from pathlib import Path
import math
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
W,H=960,540
FPS=20


def road_scene(shift: float=0., frame_index: int=0):
    img=np.zeros((H,W,3),np.uint8)  # RGB
    yy=np.arange(H).reshape(H,1)
    # Hazy blue sky and warm green roadside
    sky=np.zeros((H,1,3),np.float32)
    sky[:,:,0]=np.clip(44 + yy*0.20,0,255)
    sky[:,:,1]=np.clip(89 + yy*0.24,0,255)
    sky[:,:,2]=np.clip(133 + yy*0.23,0,255)
    img[:]=np.repeat(sky.astype(np.uint8),W,axis=1)
    horizon=303
    cv2.rectangle(img,(0,horizon),(W,H),(43,91,49),-1)
    rng=np.random.default_rng(93)
    # Distant evergreen silhouettes
    for x in range(-10,W+30,34):
        ht=int(rng.integers(18,43))
        cv2.circle(img,(x,horizon+5),ht,(36,76,46),-1)
    # Sunny horizon glow
    cv2.rectangle(img,(0,horizon-11),(W,horizon-5),(123,155,148),-1)
    perspective=.38
    t_center=int(W//2+shift*perspective)
    bottom_center=int(W//2+shift)
    top_y=306
    # Road shoulder / bitumen trapezoids
    outer=np.array([[t_center-90,top_y],[t_center+90,top_y],[bottom_center+395,H],[bottom_center-395,H]],np.int32)
    road=np.array([[t_center-60,top_y],[t_center+60,top_y],[bottom_center+315,H],[bottom_center-315,H]],np.int32)
    cv2.fillPoly(img,[outer],(102,101,83))
    cv2.fillPoly(img,[road],(66,70,73))
    # Asphalt surface faint lines
    for y in range(horizon+35,H,13):
        alpha=(y-top_y)/max(1,(H-top_y))
        cx=(1-alpha)*t_center+alpha*bottom_center
        half=(1-alpha)*60+alpha*315
        noise=int(8*math.sin(y*.08+frame_index*.16))
        cv2.line(img,(int(cx-half+30),y),(int(cx+half-20),y),(66+noise,71+noise,73+noise),1)
    # Edge boundaries: long and sufficiently slanted for the baseline Hough detector
    for side in [-1,1]:
        p0=(int(t_center + side*60),top_y)
        p1=(int(bottom_center + side*315),H)
        cv2.line(img,p0,p1,(244,241,224),12,cv2.LINE_AA)
        cv2.line(img,p0,p1,(252,251,242),4,cv2.LINE_AA)
    # Centre dashed marks animated with advancing frame index
    for k in range(9):
        depth=((k*.13+frame_index*.016)%1.0)
        d0=depth**1.8
        d1=min(1,depth+.065)**1.8
        y0=int(top_y+(H-top_y)*d0)
        y1=int(top_y+(H-top_y)*d1)
        x0=int(t_center+(bottom_center-t_center)*d0)
        x1=int(t_center+(bottom_center-t_center)*d1)
        thickness=max(2,int(3+7*d0))
        cv2.line(img,(x0,y0),(x1,y1),(239,194,75),thickness,cv2.LINE_AA)
    # Grassland texture, outside road
    for x in range(15,W,27):
        ry=int(horizon+30+(x*67)%160)
        val=int(14+((x*7)%21))
        cv2.line(img,(x,ry),(x+7,ry-4),(43+val,86+val,43+val),2,cv2.LINE_AA)
    # small low-contrast clouds
    cv2.ellipse(img,(220,126),(110,18),0,0,360,(108,145,170),-1,cv2.LINE_AA)
    cv2.ellipse(img,(705,183),(130,13),0,0,360,(115,145,157),-1,cv2.LINE_AA)
    return img


def main():
    HERE.mkdir(exist_ok=True)
    presets=[('straight',0),('left',-105),('right',105)]
    for name,shift in presets:
        image=road_scene(shift)
        cv2.imwrite(str(HERE/f'road-{name}.jpg'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY),88])
    writer=cv2.VideoWriter(str(HERE/'road-sample.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
    if not writer.isOpened():
        raise RuntimeError('MP4 codec is unavailable')
    for i in range(160):
        shift=107*math.sin(2*math.pi*(i/160)-math.pi/2)
        writer.write(cv2.cvtColor(road_scene(shift,i),cv2.COLOR_RGB2BGR))
    writer.release()
    print('Generated synthetic road image examples and 8-second source video')

if __name__=='__main__':
    main()

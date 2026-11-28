import numpy as np, math, random
from PIL import Image, ImageFilter, ImageDraw
W,H=600,600
rng=np.random.default_rng(7);random.seed(7)
Y,X=np.mgrid[0:H,0:W].astype(np.float32)
def grad(stops,t):
  t=np.clip(t,0,1);out=np.zeros(t.shape+(3,),np.float32)
  for i in range(len(stops)-1):
    a,ca=stops[i];b,cb=stops[i+1];m=(t>=a)&(t<=b);u=((t-a)/(b-a+1e-6))[...,None]
    out[m]=(np.array(ca)*(1-u)+np.array(cb)*u)[m]
  return out
def noise(scale,oct=4,seed=0):
  r=np.random.default_rng(seed);acc=np.zeros((H,W),np.float32);amp=1;tot=0
  for o in range(oct):
    s=max(2,int(scale/(2**o)));g=r.random((H//s+2,W//s+2)).astype(np.float32)
    im=Image.fromarray((g*255).astype(np.uint8)).resize((W+2*s,H+2*s),Image.BICUBIC)
    acc+=np.asarray(im,np.float32)[:H,:W]/255*amp;tot+=amp;amp*=.5
  return acc/tot
def finish(a,name,grain=10,vig=.35,warm=0,leak=None,blur=.6):
  a=np.clip(a,0,255)
  d=np.sqrt(((X-W/2)/(W/2))**2+((Y-H/2)/(H/2))**2);a*=(1-vig*np.clip(d-.35,0,1)**1.6)[...,None]
  if warm:a[...,0]+=warm;a[...,2]-=warm*.6
  if leak:cx,cy,col,s=leak;l=np.exp(-(((X-cx)**2+(Y-cy)**2)/(2*s*s)));a+=l[...,None]*np.array(col)
  a+=rng.normal(0,grain,(H,W,1))
  # film curve: lift blacks
  a=a*.92+14
  im=Image.fromarray(np.clip(a,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
  im.save(f'gen/{name}.jpg',quality=86)
def bokeh(a,n,cols,rmin,rmax,ymin=0,ymax=H,alpha=(.25,.7),blur=2,seed=1):
  r=random.Random(seed);L=Image.new('RGB',(W,H),(0,0,0));d=ImageDraw.Draw(L)
  for i in range(n):
    x=r.uniform(-20,W+20);y=r.uniform(ymin,ymax);rr=r.uniform(rmin,rmax);c=r.choice(cols);k=r.uniform(*alpha)
    d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=tuple(int(v*k) for v in c))
  L=L.filter(ImageFilter.GaussianBlur(blur));return a+np.asarray(L,np.float32)
# 1 海边日落 (写实)
t=Y/H;sky=grad([(0,(70,80,140)),(.55,(190,120,150)),(.9,(250,170,110)),(1,(255,200,140))],t/ .52)
sea=grad([(0,(200,130,110)),(.3,(90,80,110)),(1,(30,35,60))],(Y-H*.52)/(H*.48))
a=np.where((Y<H*.52)[...,None],sky,sea)
sun=np.exp(-(((X-W*.62)**2+(Y-H*.49)**2)/(2*22**2)));glow=np.exp(-(((X-W*.62)**2+((Y-H*.49)*1.4)**2)/(2*120**2)))
a+=(sun*240)[...,None]*np.array([1,.85,.6])+(glow*90)[...,None]*np.array([1,.7,.45])
rowr=np.random.default_rng(3).random(H)[:,None];wob=np.sin(X/7+rowr*40)*.5+.5;ref=np.exp(-((X-W*.62)**2)/(2*(26+np.abs(Y-H*.52)*.4)**2))*(Y>H*.52)*((rowr*.6+wob*.6)>.75)
a+=(ref*200)[...,None]*np.array([1,.8,.55])
cl=noise(140,5,4);m=np.clip((cl-.55)*4,0,1)*(Y<H*.4);a=a*(1-m[...,None]*.5)+m[...,None]*np.array([230,150,150])*.5
finish(a,'sunset',warm=6,leak=(0,0,(60,25,0),160))
# 2 城市夜景 bokeh (写实)
a=grad([(0,(12,16,40)),(.6,(30,30,60)),(1,(20,18,30))],Y/H)
a=bokeh(a,140,[(255,190,90),(255,140,80),(120,170,255),(255,220,160),(230,120,160)],6,30,H*.15,H*.95,blur=3,seed=2)
a=bokeh(a,60,[(255,210,140),(150,190,255)],18,48,0,H,alpha=(.12,.3),blur=6,seed=3)
finish(a,'citynight',grain=12,vig=.5)
# 3 雾山湖 (写实)
a=grad([(0,(205,212,222)),(.55,(235,226,222)),(1,(200,205,212))],Y/H)
for k,(base,amp,col,sd) in enumerate([(.42,.12,(170,178,195),1),(.5,.1,(130,140,160),2),(.58,.08,(85,98,112),3),(.64,.06,(52,64,72),4)]):
  r=np.random.default_rng(sd);xs=np.linspace(0,1,W);ridge=np.zeros(W)
  for f in [2,5,11,23]:ridge+=np.sin(xs*f*math.pi+r.random()*6)*amp/f*3
  ridge=(base-ridge*.5)*H;m=(Y>ridge[None,:])&(Y<H*.7)
  a[m]=a[m]*.25+np.array(col)*.75
mist=noise(80,4,9);a=a*(1-(mist*.3*(np.abs(Y-H*.6)<60))[...,None])+(mist*.3*(np.abs(Y-H*.6)<60))[...,None]*240
lake=Y>H*.7;flip=a[np.clip((2*H*.7-Y).astype(int),0,H-1),X.astype(int)];a[lake]=flip[lake]*.8+np.array([150,160,175])*.2
a[(np.abs(Y-H*.7)<1.2)]+=25
finish(a,'mistlake',grain=8,vig=.25,blur=.8)
# 4 星空 (写实)
a=grad([(0,(8,10,28)),(.7,(30,30,70)),(1,(70,60,90))],Y/H)
mw=np.exp(-((Y-(H*.9-X*.8))**2)/(2*70**2));nn=noise(30,5,11);a+=(mw*nn*120)[...,None]*np.array([.8,.75,1])
st=rng.random((H,W));a[st>.9965]+=rng.uniform(80,255,(int((st>.9965).sum()),1))
r=np.random.default_rng(5);xs=np.linspace(0,1,W);ridge=H*.78-np.cumsum(r.normal(0,1.6,W));ridge=np.convolve(ridge,np.ones(15)/15,'same')
a[Y>ridge[None,:]]=np.array([10,10,16])
finish(a,'stars',grain=9,vig=.4,blur=.5)
# 5 樱花 bokeh 午后 (写实)
a=grad([(0,(200,220,240)),(1,(245,235,240))],Y/H)
a=bokeh(a,90,[(255,190,210),(250,170,195),(255,230,235)],14,40,0,H,alpha=(.25,.55),blur=7,seed=6)
L=Image.new('L',(W,H),0);d=ImageDraw.Draw(L)
def br(x,y,ang,ln,w,dep):
  if dep==0 or ln<8:return
  x2=x+math.cos(ang)*ln;y2=y+math.sin(ang)*ln;d.line((x,y,x2,y2),fill=255,width=int(w))
  for s in (-.5,.4):br(x2,y2,ang+s+random.uniform(-.2,.2),ln*.72,w*.65,dep-1)
br(W+10,70,math.pi*.93,190,26,7);L=L.filter(ImageFilter.GaussianBlur(1.2));m=np.asarray(L,np.float32)[...,None]/255
a=a*(1-m)+m*np.array([70,50,55])
a=bokeh(a,160,[(255,200,215),(255,240,245),(245,160,190)],3,8,0,H*.75,alpha=(.6,1),blur=1,seed=8)
finish(a,'sakura',grain=7,vig=.2,blur=.4,leak=(W,0,(40,20,10),200))
# 6 粉色月亮海 (不真实)
a=grad([(0,(120,110,190)),(.45,(240,170,200)),(.55,(250,200,210)),(1,(120,120,190))],Y/H)
moon=((X-W*.5)**2+(Y-H*.35)**2)<95**2;mg=np.exp(-(((X-W*.5)**2+(Y-H*.35)**2)/(2*150**2)))
a+=(mg*70)[...,None]*np.array([1,.8,.9]);cr=noise(25,4,12);a[moon]=(np.array([255,225,235])*(.88+cr[...,None]*.15))[moon]
sea=Y>H*.6;a[sea]=a[sea]*.6+np.array([90,90,170])*.4
rowr=np.random.default_rng(13).random(H)[:,None];rf=np.exp(-((X-W*.5)**2)/(2*(40+(Y-H*.6)*.3)**2))*sea*((rowr*.6+(np.sin(X/6+rowr*50)*.5+.5)*.6)>.78);a+=(rf*130)[...,None]*np.array([1,.85,.9])
a=bokeh(a,120,[(255,255,255),(255,220,240)],1,2.4,0,H*.55,alpha=(.5,1),blur=.6,seed=9)
finish(a,'pinkmoon',grain=6,vig=.3,blur=.5)
# 7 粉云热气球 (不真实)
a=grad([(0,(160,180,235)),(.6,(245,200,215)),(1,(255,225,215))],Y/H)
cl=noise(110,6,14);m=np.clip((cl-.48)*3,0,1)*np.clip((Y-H*.45)/(H*.3),0,1);a=a*(1-m[...,None])+m[...,None]*np.array([255,236,240])
for (cx,cy,rr,c1,c2) in [(300,210,95,(240,140,160),(250,200,140)),(140,330,45,(150,140,220),(230,190,240)),(470,370,32,(250,190,120),(250,150,160))]:
  bm=(((X-cx)/rr)**2+((Y-cy)/(rr*1.12))**2)<1;st=(np.sin((X-cx)/rr*7)>0)
  col=np.where(st[...,None],np.array(c1),np.array(c2)).astype(np.float32);sh=(1-((X-cx+rr*.4)/(rr*2))**2*.5)[...,None]
  a[bm]=(col*np.clip(sh,.6,1.1))[bm]
  bk=(np.abs(X-cx)<rr*.15)&(Y>cy+rr*1.35)&(Y<cy+rr*1.55);a[bk]=np.array([120,85,60])
  for sx in (-.5,.5):
    ln=(np.abs(X-(cx+sx*rr*(1-(Y-cy-rr*1.0)/(rr*.4)*0.7)))<1.2)&(Y>cy+rr*.95)&(Y<cy+rr*1.36);a[ln]=np.array([110,90,80])
a=bokeh(a,40,[(255,255,255)],1,2,0,H*.5,blur=.5,seed=15)
finish(a,'balloon',grain=6,vig=.2,blur=.6)
# 8 雪夜窗 (半写实)
a=grad([(0,(25,35,70)),(1,(60,70,110))],Y/H)
a=bokeh(a,70,[(255,200,120)],6,22,H*.55,H*.9,alpha=(.3,.7),blur=5,seed=16)
a=bokeh(a,500,[(255,255,255)],1,3.5,0,H,alpha=(.5,.95),blur=.8,seed=17)
fr=((np.abs(X-W/2)<10)|(np.abs(Y-H/2)<10)|(X<26)|(X>W-26)|(Y<26)|(Y>H-26))
a[fr]=np.array([235,215,190])*.9;cond=np.exp(-(np.minimum.reduce([X,W-X,Y,H-Y])-26)/40)*(~fr);a+=(cond*80)[...,None]
finish(a,'snowwindow',grain=8,vig=.35,warm=5)
# 9 咖啡馆 (写实)
a=grad([(0,(60,38,25)),(1,(110,75,50))],Y/H)
a=bokeh(a,60,[(255,190,110),(255,220,150),(255,160,90)],15,45,0,H*.6,alpha=(.25,.6),blur=8,seed=18)
L=Image.new('L',(W,H),0);d=ImageDraw.Draw(L);d.ellipse((200,430,420,470),fill=200);d.polygon([(215,330),(405,330),(380,450),(240,450)],fill=255);d.ellipse((395,350,455,410),outline=255,width=14)
L=L.filter(ImageFilter.GaussianBlur(1.5));m=np.asarray(L,np.float32)[...,None]/255
cup=grad([(0,(245,240,230)),(1,(200,185,170))],(X-200)/220);a=a*(1-m)+cup*m
a[(Y>480)]=a[(Y>480)]*.5+np.array([90,60,40])*.5
finish(a,'cafe',grain=10,vig=.45,warm=8)
print('ok')

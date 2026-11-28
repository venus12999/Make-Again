/* ===== 纸团（来自你之前做的「纸团展开」，Three.js 三角折面 + 顶点插值） ===== */
const PaperKit=(()=>{const T=window.THREE;if(!T)return null;
const CONFIG={sheetW:3.3,sheetH:4.1,speeds:{fast:[0.9,1.0]},speed:'fast',stopMotion:{cycle:0.083,hold:0.05},spread:0.3,flatCrease:0.13,detail:1.0};
const STYLES=[
  { name:'手写信：紧实小圆球，细碎褶皱；从中心向外展开，揉回时从右上角先收',
    R:0.86, sx:1.0, sy:1.0, sz:0.85, twist:0.6, res:[34,42], crinkle:1.0, bigPlanes:10, smallPlanes:0, bigK:[0.1,0.2], bigW:[0.05,0.11], noise:0.14,
    segs:70, segBig:0.7, foldAngle:null, fine:80,
    open:{type:'center', steps:7}, close:{type:'corner', dir:[1,1], steps:6}, rotSign:1 },
  { name:'照片：松散大团，大块深折面；从左下角掀开，揉回时边缘先往里收',
    R:0.98, sx:0.95, sy:1.05, sz:0.72, twist:-1.4, res:[31,39], crinkle:1.1, bigPlanes:8, smallPlanes:0, bigK:[0.18,0.34], bigW:[0.08,0.18], noise:0.22,
    segs:40, segBig:0.85, foldAngle:null, fine:45,
    open:{type:'corner', dir:[-1,-1], steps:6}, close:{type:'edges', steps:5}, rotSign:-1 },
  { name:'打字信：拧过的扁长条，横向长折痕；像卷轴从中间往上下摊开，揉回时外圈一瓣瓣收拢',
    R:0.92, sx:1.3, sy:0.78, sz:0.75, twist:2.6, res:[38,47], crinkle:0.9, bigPlanes:10, smallPlanes:0, bigK:[0.12,0.24], bigW:[0.05,0.12], noise:0.16,
    segs:55, segBig:0.75, foldAngle:0.25, fine:60,
    open:{type:'band', steps:8}, close:{type:'petal', steps:6}, rotSign:1 },
];

const TW=1024,TH=Math.round(1024*CONFIG.sheetH/CONFIG.sheetW);
const LIGHT=new T.Vector3(-0.38,0.5,0.78).normalize();let maxAniso=4;
/* ---------- 随机与噪声 ---------- */
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function hash(x,y,z,s){let h=Math.imul(x|0,374761393)^Math.imul(y|0,668265263)^Math.imul(z|0,1440662683)^Math.imul(s|0,-1640531535);h=Math.imul(h^(h>>>13),1274126177);h^=h>>>16;return(h>>>0)/4294967295;}
function vnoise(x,y,z,s){
  const xi=Math.floor(x),yi=Math.floor(y),zi=Math.floor(z);
  const xf=x-xi,yf=y-yi,zf=z-zi;
  const u=xf*xf*(3-2*xf),v=yf*yf*(3-2*yf),w=zf*zf*(3-2*zf);
  const L=(a,b,t)=>a+(b-a)*t, c=(i,j,k)=>hash(xi+i,yi+j,zi+k,s);
  return L(L(L(c(0,0,0),c(1,0,0),u),L(c(0,1,0),c(1,1,0),u),v),L(L(c(0,0,1),c(1,0,1),u),L(c(0,1,1),c(1,1,1),u),v),w)*2-1;
}

/* ---------- 折痕：直线折 + 局部短折痕 ---------- */
function makeCreases(rand,W,H,st){
  const folds=[],segs=[];
  const ang=()=>st.foldAngle==null?rand()*Math.PI:st.foldAngle+(rand()-.5)*0.7;
  for(let i=0;i<8;i++){const a=ang();folds.push({x:(rand()-.5)*W*.7,y:(rand()-.5)*H*.7,nx:Math.cos(a),ny:Math.sin(a),k:(rand()<.5?-1:1)*(0.08+rand()*0.14)});}
  for(let i=0;i<Math.round(st.segs*0.4);i++){const a=ang(),big=rand()<st.segBig;segs.push({x:(rand()-.5)*W,y:(rand()-.5)*H,nx:Math.cos(a),ny:Math.sin(a),w:big?0.25+rand()*0.3:0.12+rand()*0.08,len:big?0.9+rand()*1.6:0.4+rand()*0.6,k:(rand()<.5?-1:1)*(big?0.12+rand()*0.22:0.05+rand()*0.1)});}
  return {folds,segs};
}
function creaseField(cr,x,y){
  let z=0;
  for(const f of cr.folds){z+=f.k*Math.abs((x-f.x)*f.nx+(y-f.y)*f.ny);}
  for(const g of cr.segs){
    const dx=x-g.x,dy=y-g.y,d=dx*g.nx+dy*g.ny,t=-dx*g.ny+dy*g.nx;
    const a=1-Math.abs(d)/g.w,b=1-Math.abs(t)/g.len;
    if(a>0&&b>0)z+=g.k*a*Math.min(1,b*2.5);
  }
  return z;
}

/* ---------- 细褶皱法线贴图：上百条随机直线折棱叠加成高度场 ---------- */
function detailNormalTex(seed,fine){
  const w=512,h=Math.round(512*CONFIG.sheetH/CONFIG.sheetW),hf=new Float32Array(w*h),rand=mulberry(seed);
  const addSeg=(cx,cy,ang,L,wd,amp,soft)=>{
    const nx=Math.cos(ang),ny=Math.sin(ang),r=L+wd;
    const x0=Math.max(0,Math.floor(cx-r)),x1=Math.min(w-1,Math.ceil(cx+r)),y0=Math.max(0,Math.floor(cy-r)),y1=Math.min(h-1,Math.ceil(cy+r));
    for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
      const dx=x-cx,dy=y-cy,d=Math.abs(dx*nx+dy*ny);if(d>=wd)continue;
      const t=Math.abs(-dx*ny+dy*nx);if(t>=L)continue;
      hf[y*w+x]+=amp*(1-d/wd)*Math.min(1,(L-t)/(L*soft));
    }
  };
  for(let i=0;i<8;i++)addSeg(rand()*w,rand()*h,rand()*Math.PI,w*1.4,30+rand()*40,(rand()<.5?-1:1)*(8+rand()*10),0.2);
  for(let i=0;i<70;i++)addSeg(rand()*w,rand()*h,rand()*Math.PI,40+rand()*170,8+rand()*16,(rand()<.5?-1:1)*(3+rand()*5),0.35);
  for(let i=0;i<fine;i++)addSeg(rand()*w,rand()*h,rand()*Math.PI,8+rand()*45,2+rand()*5,(rand()<.5?-1:1)*(0.8+rand()*1.6),0.45);
  const c=document.createElement('canvas');c.width=w;c.height=h;
  const ctx=c.getContext('2d'),id=ctx.createImageData(w,h),d=id.data,k=0.55;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const xl=hf[y*w+Math.max(0,x-1)],xr=hf[y*w+Math.min(w-1,x+1)],yu=hf[Math.max(0,y-1)*w+x],yd=hf[Math.min(h-1,y+1)*w+x];
    let nx=-(xr-xl)*k,ny=(yd-yu)*k,nz=1;const l=Math.hypot(nx,ny,nz);nx/=l;ny/=l;nz/=l;
    const i=(y*w+x)*4;d[i]=(nx*.5+.5)*255;d[i+1]=(ny*.5+.5)*255;d[i+2]=(nz*.5+.5)*255;d[i+3]=255;
  }
  ctx.putImageData(id,0,0);
  const t=new T.CanvasTexture(c);t.anisotropy=8;return t;
}

const paperVS=`
attribute float ao;
varying vec2 vUv; varying vec3 vPos; varying float vAo;
void main(){ vUv=uv; vAo=ao; vec4 wp=modelMatrix*vec4(position,1.0); vPos=wp.xyz; gl_Position=projectionMatrix*viewMatrix*wp; }`;
const paperFS=`
uniform sampler2D map; uniform sampler2D detail; uniform vec3 backCol; uniform float crumple; uniform float wrinkle; uniform float detailK; uniform float flatK; uniform vec3 L;
varying vec2 vUv; varying vec3 vPos; varying float vAo;
void main(){
  vec3 q0=dFdx(vPos), q1=dFdy(vPos);
  vec3 N=normalize(cross(q0,q1));
  if(N.z<0.0) N=-N;
  // 细褶皱：法线贴图扰动（切线空间来自屏幕导数）
  vec2 st0=dFdx(vUv), st1=dFdy(vUv);
  vec3 q1p=cross(q1,N), q0p=cross(N,q0);
  vec3 Tt=q1p*st0.x+q0p*st1.x, Bt=q1p*st0.y+q0p*st1.y;
  float det=max(dot(Tt,Tt),dot(Bt,Bt)); float sc=det==0.0?0.0:inversesqrt(det);
  vec3 mN=texture2D(detail,vUv).xyz*2.0-1.0;
  mN.xy*=flatK*(1.0-crumple)*detailK;
  vec3 n=normalize(Tt*(mN.x*sc)+Bt*(mN.y*sc)+N*mN.z);
  if(n.z<0.0) n=-n;
  float diff=max(dot(n,L),0.0);
  vec3 base=gl_FrontFacing ? texture2D(map,vUv).rgb : backCol;
  float ao=mix(1.0,vAo,crumple);
  // 柔和的“包裹”漫反射 + 阴面压暗，折面对比更像真纸
  float shade=(0.52+0.6*diff)*ao;
  gl_FragColor=vec4(base*shade,1.0);
}`;


/* 投影阴影：把纸张真实形状沿光线方向压到桌面上，影子能看出纸的轮廓和折角 */
const shadowVS=`
uniform float base; uniform float deskZ; uniform vec2 off; uniform float grow;
void main(){
  vec4 wp=modelMatrix*vec4(position,1.0);
  float h=max(0.0,position.z+base);
  vec2 g=position.xy; float gl=length(g); g=gl>1e-4?g/gl:vec2(0.0);
  wp.xy+=off*h+g*grow;
  wp.z=deskZ;
  gl_Position=projectionMatrix*viewMatrix*wp;
}`;
const shadowFS=`uniform vec3 col; void main(){ gl_FragColor=vec4(col,1.0); }`;
function shadowMat(col,off,grow,deskZ){
  return new T.ShaderMaterial({uniforms:{col:{value:new T.Color(col)},off:{value:new T.Vector2(off[0],off[1])},grow:{value:grow},base:{value:1},deskZ:{value:deskZ}},
    vertexShader:shadowVS,fragmentShader:shadowFS,side:T.DoubleSide});
}

let orderCounter=0;
const lerpR=(rand,[a,b])=>a+(b-a)*rand();
// 纸面各处的先后顺序：值越小越先动
function patternField(F,n,W,H,pat,isClose){
  const d=new Float32Array(n);let mn=1e9,mx=-1e9;
  for(let k=0;k<n;k++){
    const nx=F[k*3]/(W/2),ny=F[k*3+1]/(H/2),rad=Math.hypot(nx,ny)/Math.SQRT2;
    let v;
    if(pat.type==='center')v=rad;
    else if(pat.type==='edges')v=-rad;
    else if(pat.type==='corner')v=-(nx*pat.dir[0]+ny*pat.dir[1]);
    else if(pat.type==='band')v=Math.abs(ny)+0.15*Math.abs(nx);
    else v=-rad*(0.8+0.35*Math.sin(2*Math.atan2(ny,nx)+1.0)); // petal
    d[k]=v;mn=Math.min(mn,v);mx=Math.max(mx,v);
  }
  for(let k=0;k<n;k++){const t=(d[k]-mn)/(mx-mn||1);d[k]=isClose?1-t:t;}
  return d;
}
class Paper{
  constructor(scene,idx,seed,draw){
    this.idx=idx;this.seed=seed;this.draw=draw;const st=this.st=STYLES[idx];
    const rand=mulberry(seed);
    const {sheetW:W,sheetH:H}=CONFIG,[NX,NY]=st.res,R=st.R;
    this.cr=makeCreases(mulberry(seed*7+3),W,H,st);
    this.cv=document.createElement('canvas');this.cv.width=TW;this.cv.height=TH;
    this.tex=new T.CanvasTexture(this.cv);this.tex.anisotropy=maxAniso;
    this.redraw();
    const n=(NX+1)*(NY+1);
    const pos=new Float32Array(n*3),uv=new Float32Array(n*2),ao=new Float32Array(n);
    this.B=new Float32Array(n*3);this.F=new Float32Array(n*3);this.Wz=new Float32Array(n);this.J=new Float32Array(n*3);this.AO=new Float32Array(n);
    const field=new Float32Array(n);
    for(let j=0;j<=NY;j++)for(let i=0;i<=NX;i++){
      const k=j*(NX+1)+i;
      let u=i/NX,v=j/NY;
      if(i>0&&i<NX)u+=(rand()-.5)*0.6/NX;
      if(j>0&&j<NY)v+=(rand()-.5)*0.6/NY;
      const x=(u-.5)*W,y=(v-.5)*H;
      uv[k*2]=u;uv[k*2+1]=v;
      this.F[k*3]=x;this.F[k*3+1]=y;
      field[k]=creaseField(this.cr,x,y);
    }
    let mean=0;for(let k=0;k<n;k++)mean+=field[k];mean/=n;
    let mx=1e-6;for(let k=0;k<n;k++){field[k]-=mean;mx=Math.max(mx,Math.abs(field[k]));}
    for(let k=0;k<n;k++){const f=field[k]/mx;this.Wz[k]=f*0.16;this.F[k*3+2]=f*0.03*CONFIG.flatCrease/0.13*0.35;}
    // 展开中途的硬折面：每个顶点一个随机三维错位，边缘更皱
    for(let k=0;k<n;k++){
      const nx=this.F[k*3]/(W/2),ny=this.F[k*3+1]/(H/2),edge=0.55+0.65*Math.min(1,Math.hypot(nx,ny)/Math.SQRT2);
      this.J[k*3]=(rand()-.5)*0.035*st.crinkle*edge;this.J[k*3+1]=(rand()-.5)*0.035*st.crinkle*edge;this.J[k*3+2]=(rand()-.5)*0.11*st.crinkle*edge;
    }
    // 纸团形状
    const planes=[];for(let i=0;i<st.bigPlanes+st.smallPlanes;i++){const z=rand()*2-1,a=rand()*Math.PI*2,s=Math.sqrt(1-z*z),big=i<st.bigPlanes;
      planes.push({x:s*Math.cos(a),y:s*Math.sin(a),z,o:(rand()-.5)*1.4,w:big?lerpR(rand,st.bigW):0.04+rand()*0.06,k:(rand()<.5?-1:1)*(big?lerpR(rand,st.bigK):0.04+rand()*0.08)});}
    const twist=st.twist*(0.8+0.4*rand());
    const tA=0.7+rand()*0.25,tB=(rand()-.5)*0.7,cA=Math.cos(tA),sA=Math.sin(tA),cB=Math.cos(tB),sB=Math.sin(tB);
    for(let k=0;k<n;k++){
      const nx=this.F[k*3]/(W/2),ny=this.F[k*3+1]/(H/2);
      const rad=Math.min(1,Math.hypot(nx,ny)/Math.SQRT2);
      const th=Math.atan2(ny,nx)+twist*rad+0.8*vnoise(nx*1.6,ny*1.6,1.3,seed)+0.35*vnoise(nx*4.1,ny*4.1,3.7,seed);
      const ph=Math.min(Math.PI*0.995,rad*Math.PI*0.93+0.45*vnoise(nx*2.2,ny*2.2,7.1,seed)*Math.min(1,rad*3)+0.18*vnoise(nx*5,ny*5,2.2,seed));
      const ex=Math.sin(ph)*Math.cos(th),ey=Math.sin(ph)*Math.sin(th),ez=Math.cos(ph);
      const dx=ex*cA+ez*sA,dz0=-ex*sA+ez*cA;
      const dy=ey*cB-dz0*sB,dz=ey*sB+dz0*cB;
      let r=1+st.noise*vnoise(dx*1.5+3,dy*1.5,dz*1.5,seed+1)+st.noise*0.5*vnoise(dx*3.4,dy*3.4,dz*3.4+5,seed+2);
      let c=0;for(const p of planes){const a=1-Math.abs(dx*p.x+dy*p.y+dz*p.z-p.o)/p.w;if(a>0)c+=p.k*a;}
      r+=c*0.3+(rand()-.5)*0.26*st.crinkle;
      const tj=0.06*st.crinkle;
      this.B[k*3]=dx*r*R*st.sx+(rand()-.5)*tj;this.B[k*3+1]=dy*r*R*st.sy+(rand()-.5)*tj;this.B[k*3+2]=dz*r*R*st.sz;
      this.AO[k]=Math.min(1,Math.max(0.72,0.86+0.4*(r-0.9)+0.08*dz));
    }
    this.Dopen=patternField(this.F,n,W,H,st.open,false);
    this.Dclose=patternField(this.F,n,W,H,st.close,true);
    this.D=this.Dopen;
    pos.set(this.B);ao.set(this.AO);
    const idxArr=[];
    for(let j=0;j<NY;j++)for(let i=0;i<NX;i++){
      const a=j*(NX+1)+i,b=a+1,c=a+NX+1,d=c+1;
      if(rand()<.5)idxArr.push(a,b,d,a,d,c);else idxArr.push(a,b,c,b,d,c);
    }
    const g=new T.BufferGeometry();
    g.setAttribute('position',new T.BufferAttribute(pos,3).setUsage(T.DynamicDrawUsage));
    g.setAttribute('uv',new T.BufferAttribute(uv,2));
    g.setAttribute('ao',new T.BufferAttribute(ao,1));
    g.setIndex(idxArr);
    this.geo=g;
    this.mat=new T.ShaderMaterial({uniforms:{map:{value:this.tex},detail:{value:detailNormalTex(seed*13+5,st.fine)},wrinkle:{value:1},detailK:{value:CONFIG.detail},flatK:{value:CONFIG.flatCrease},backCol:{value:new T.Color(0.955,0.948,0.93)},crumple:{value:1},L:{value:LIGHT}},
      vertexShader:paperVS,fragmentShader:paperFS,side:T.DoubleSide,extensions:{derivatives:true}});
    this.mesh=new T.Mesh(g,this.mat);this.mesh.frustumCulled=false;
    // 两层影子：外层浅（半影），内层深（本影），都是纸张真实轮廓
    this.penumbra=new T.Mesh(g,shadowMat(0xE6E0EE,[0.16,-0.22],0.03,-6.02));
    this.umbra=new T.Mesh(g,shadowMat(0xD9D1E3,[0.1,-0.14],0.0,-6.0));
    this.penumbra.frustumCulled=this.umbra.frustumCulled=false;
    this.group=new T.Group();this.group.add(this.mesh,this.penumbra,this.umbra);
    this.rot=st.rotSign*(0.2+rand()*0.35);this.wob=st.rotSign*(0.15+rand()*0.12);
    this.group.rotation.z=this.rot;
    scene.add(this.group);
    this.p=0;this.s=0;this.target=0;this.goal=0;this.anim=null;this.order=0;this.dirty=true;this.x=0;this.y=0;
    this.hitR=R*Math.max(st.sx,st.sy)*1.25;
  }
  redraw(){this.draw(this.cv.getContext('2d'),this.cr,this.seed);this.tex.needsUpdate=true;}
  setTarget(t){if(t===1&&this.target!==1)this.order=++orderCounter;this.target=t;}
  hit(wx,wy){
    const dx=wx-this.x,dy=wy-this.y;
    if(this.s>0.25){const k=0.35+0.65*this.s;return Math.abs(dx)<CONFIG.sheetW/2*k+0.05&&Math.abs(dy)<CONFIG.sheetH/2*k+0.05;}
    return Math.hypot(dx,dy)<this.hitR;
  }
  // 连续的缓动曲线；“硬纸”的顿挫感由主循环里的定格节拍产生
  startAnim(){
    const from=this.p,to=this.goal,dist=Math.abs(to-from);
    if(dist<1e-4){this.anim=null;this.p=to;return;}
    const opening=to>from;
    if(!this.anim)this.D=opening?this.Dopen:this.Dclose;   // 只在静止时切换动作方式，避免跳变
    const sp=CONFIG.speeds[CONFIG.speed];
    this.anim={from,to,t:0,dur:Math.max(0.2,(opening?sp[0]:sp[1])*dist),opening};
  }
  step(dt,frozen){
    if(this.target!==this.goal){this.goal=this.target;this.startAnim();}
    if(this.anim){
      const A=this.anim;A.t+=dt;
      const k=Math.min(1,A.t/A.dur);
      // 展开：先加速后减速（中段最快）；揉回：起手更快，后段慢慢收紧
      const e=A.opening?(k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2):(1-Math.pow(1-k,2.4));
      this.p=A.from+(A.to-A.from)*e;
      if(k>=1){this.p=A.to;this.anim=null;}
      this.dirty=true;
    }
    if(frozen&&this.anim)return;      // 定格帧：画面停住，时间照走
    if(!this.dirty)return;this.dirty=false;
    const P=Math.min(1,Math.max(0,this.p)),SP=CONFIG.spread,D=this.D;
    const Pa=this.geo.attributes.position.array,B=this.B,F=this.F,Wz=this.Wz,J=this.J;
    let acc=0;
    for(let k=0,n=Wz.length;k<n;k++){
      let pv=P*(1+SP)-D[k]*SP;pv=pv<0?0:pv>1?1:pv;
      const s=pv*pv*(3-2*pv),c=1-s,bell=Math.sin(Math.PI*pv),i=k*3;acc+=s;
      const cr=c>0?Math.pow(c,1.3)*bell*1.6:0;
      Pa[i]=B[i]*c+F[i]*s+J[i]*cr;
      Pa[i+1]=B[i+1]*c+F[i+1]*s+J[i+1]*cr;
      Pa[i+2]=B[i+2]*c+F[i+2]*s+(Wz[k]*bell+J[i+2]*cr);
    }
    this.geo.attributes.position.needsUpdate=true;
    // 用纸面“平均打开程度”驱动阴影、旋转和层级，避免纸还没展开就出现方形阴影
    const S=acc/Wz.length,C=1-S,bellG=Math.sin(Math.PI*S);
    this.s=S;
    this.mat.uniforms.crumple.value=C;
    this.mat.uniforms.wrinkle.value=Math.min(1,Math.max(C,bellG*0.5));
    this.group.rotation.z=this.rot*C+this.wob*bellG*0.12;
    this.group.position.set(this.x,this.y,P>0?2.2+1.4*S+this.order*0.001:0);
    // 影子高度：纸团底部贴着桌面；展开中途纸张离桌面高一些；平铺时只浮起一点点
    const zb=this.st.R*this.st.sz*1.05, hb=zb*C+0.22*S+0.45*bellG;
    this.umbra.material.uniforms.base.value=hb;this.penumbra.material.uniforms.base.value=hb;
  }
}

/* 舞台：一块 canvas，一台正交相机，可以放好几张纸 */
function Stage(cv,ppu){let R;try{R=new T.WebGLRenderer({canvas:cv,antialias:true,alpha:true})}catch(e){return null}
  R.setPixelRatio(Math.min(devicePixelRatio||1,2));R.setClearColor(0,0);maxAniso=R.capabilities.getMaxAnisotropy();
  const scene=new T.Scene(),cam=new T.OrthographicCamera(-1,1,1,-1,.1,60);cam.position.set(0,0,30);cam.lookAt(0,0,0);
  let W=1,H=1;const S={papers:[],ppu:ppu||72,scene};
  S.resize=()=>{W=cv.clientWidth||1;H=cv.clientHeight||1;R.setSize(W,H,false);const vw=W/S.ppu,vh=H/S.ppu;cam.left=-vw/2;cam.right=vw/2;cam.top=vh/2;cam.bottom=-vh/2;cam.updateProjectionMatrix();S.papers.forEach(p=>p.dirty=true)};
  S.renderNow=()=>R.render(scene,cam);
  S.toWorld=(px,py)=>[(px-W/2)/S.ppu,(H/2-py)/S.ppu];
  S.add=(draw,style,seed,p0)=>{const p=new Paper(scene,style,seed,draw);p.p=p0||0;p.target=p.goal=p0||0;p.dirty=true;S.papers.push(p);return p};
  S.remove=p=>{scene.remove(p.group);p.geo.dispose();p.mat.dispose();p.tex.dispose();S.papers=S.papers.filter(x=>x!==p)};
  let last=performance.now(),raf=0,alive=true;
  const frame=n=>{if(!alive)return;const dt=Math.min(.05,(n-last)/1000);last=n;const sm=CONFIG.stopMotion,fr=((n/1000)%sm.cycle)<sm.hold;
    S.onFrame&&S.onFrame(dt);S.papers.forEach(p=>p.step(dt,fr));R.render(scene,cam);raf=requestAnimationFrame(frame)};
  S.resize();raf=requestAnimationFrame(frame);
  S.destroy=()=>{alive=false;cancelAnimationFrame(raf);S.papers.slice().forEach(S.remove);R.dispose();try{R.forceContextLoss()}catch(e){}};
  return S}
/* 把写下的话画成一张信纸 */
const HAND='"Long Cang","Ma Shan Zheng","Kaiti SC","STKaiti",cursive';
function noteDraw(text,date,to){return (ctx,cr,seed)=>{const rand=mulberry(seed);ctx.fillStyle='#f2eee6';ctx.fillRect(0,0,TW,TH);
  for(let i=0;i<10;i++){const x=rand()*TW,y=rand()*TH,r=120+rand()*320,g=ctx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,`rgba(150,130,100,${.025+rand()*.03})`);g.addColorStop(1,'rgba(150,130,100,0)');ctx.fillStyle=g;ctx.fillRect(0,0,TW,TH)}
  const ink='#2c3352';ctx.fillStyle=ink;ctx.textBaseline='alphabetic';
  ctx.font=`40px ${HAND}`;ctx.textAlign='right';ctx.fillText(date||'',904,150);ctx.textAlign='left';
  ctx.font=`54px ${HAND}`;ctx.fillText(to||'树洞：',110,250);
  ctx.font=`50px ${HAND}`;let y=350;const mw=TW-230;
  (text||'').split('\n').forEach(par=>{let line='';for(const ch of par){if(ctx.measureText(line+ch).width>mw){ctx.fillText(line,110,y);y+=80;line=ch}else line+=ch}ctx.fillText(line,110,y);y+=80});
  const id=ctx.getImageData(0,0,TW,TH),d=id.data;for(let i=0;i<d.length;i+=4){const n=(rand()-.5)*9;d[i]+=n;d[i+1]+=n;d[i+2]+=n}ctx.putImageData(id,0,0)}}
return {Stage,noteDraw,CONFIG}})();

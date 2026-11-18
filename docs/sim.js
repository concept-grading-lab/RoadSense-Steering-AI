/* RoadSense: independent, deterministic browser driving simulation.
 * This is not a trained neural network, and is never a physical-car controller.
 */
'use strict';
const $ = id => document.getElementById(id);
const road = $('road');
const ctx = road.getContext('2d');
const video = $('processedVideo');
const state = {mode:'sim', running:true, auto:true, route:'coastal', distance:0, speed:54, lateral:0, steer:0, lastTs:0, heldLeft:false, heldRight:false, raf:0};
const W=1080,H=620,HORIZON=240,MAX_DIST=175;
const clamp = (v,min,max)=>Math.min(max,Math.max(min,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const centerAt = z=>{
 if(state.route==='straight') return .65*Math.sin(z/550);
 if(state.route==='sweeps') return 2.9*Math.sin(z/115)+1.5*Math.sin(z/250);
 return 2.1*Math.sin(z/175)+.8*Math.sin(z/75);
};
const project=(z)=>{
 const ahead=Math.max(0,z-state.distance);
 const p=1-Math.sqrt(clamp(ahead/MAX_DIST,0,1));
 const x=W/2+(centerAt(z)-state.lateral)*(15+51*p);
 const y=HORIZON+(H-HORIZON)*p;
 const half=4+W*.64*p*p;
 return {x,y,half,p};
};
function poly(points,fill){ctx.fillStyle=fill;ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);for(const [x,y] of points.slice(1))ctx.lineTo(x,y);ctx.closePath();ctx.fill();}
function line(points,stroke,width){ctx.lineWidth=width;ctx.strokeStyle=stroke;ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);for(const [x,y] of points.slice(1))ctx.lineTo(x,y);ctx.stroke();}
function drawBackdrop(){
 const sky=ctx.createLinearGradient(0,0,0,HORIZON);sky.addColorStop(0,'#78bed7');sky.addColorStop(.7,'#c8e7e7');sky.addColorStop(1,'#d5e8da');ctx.fillStyle=sky;ctx.fillRect(0,0,W,HORIZON+40);
 ctx.fillStyle='#e6f4cf';ctx.beginPath();ctx.arc(W*.77,95,40,0,7);ctx.fill();
 ctx.fillStyle='#83b0a5';ctx.beginPath();ctx.moveTo(0,HORIZON+8);for(let x=0;x<=W+50;x+=30){let a=x/120+state.distance/1700;ctx.lineTo(x,212+23*Math.sin(a)+11*Math.sin(a*2.7))}ctx.lineTo(W,HORIZON+24);ctx.lineTo(0,HORIZON+24);ctx.fill();
 ctx.fillStyle='#618f86';ctx.beginPath();ctx.moveTo(0,HORIZON+24);for(let x=0;x<=W+40;x+=20){ctx.lineTo(x,235+15*Math.sin(x/91+state.distance/1600)+7*Math.sin(x/41))}ctx.lineTo(W,HORIZON+33);ctx.lineTo(0,HORIZON+33);ctx.fill();
 const land=ctx.createLinearGradient(0,HORIZON,0,H);land.addColorStop(0,'#8dba80');land.addColorStop(1,'#4e8974');ctx.fillStyle=land;ctx.fillRect(0,HORIZON,W,H-HORIZON);
}
function drawRoad(){
 // Build road in far-to-near quadrilateral strips: the scenery moves as world distance increases.
 for(let i=0;i<122;i++){
  let far=state.distance+MAX_DIST*(1-i/122),near=state.distance+MAX_DIST*(1-(i+1)/122),a=project(far),b=project(near);
  const strip=Math.floor((far+near)/31);
  poly([[a.x-a.half,a.y],[a.x+a.half,a.y],[b.x+b.half,b.y+1],[b.x-b.half,b.y+1]],strip%2?'#384c53':'#3a4e54');
  if(a.p>.035){
   const edge=Math.max(2,9*b.p);
   poly([[a.x-a.half,a.y],[a.x-a.half+edge*.6,a.y],[b.x-b.half+edge,b.y+1],[b.x-b.half,b.y+1]],'#e2e9de');
   poly([[a.x+a.half-edge*.6,a.y],[a.x+a.half,a.y],[b.x+b.half,b.y+1],[b.x+b.half-edge,b.y+1]],'#e2e9de');
   if(Math.floor(far/14)%2===0){
     for(const side of [-1,1]){
      const mid=side*.34;
      poly([[a.x+a.half*mid-1.5,a.y],[a.x+a.half*mid+1.5,a.y],[b.x+b.half*mid+Math.max(2,4*b.p),b.y+1],[b.x+b.half*mid-Math.max(2,4*b.p),b.y+1]],'#e2e0cf');
     }
   }
  }
 }
}
function tree(x,y,size,index){
 if(size<2) return;
 ctx.fillStyle='#41554e';ctx.fillRect(x-size*.09,y-size*.25,size*.18,size*.36);
 ctx.fillStyle=index%3===0?'#2e6a58':index%3===1?'#34765c':'#3b805a';
 for(let i=0;i<3;i++){let dy=i*size*.17;poly([[x,y-size*.95+dy],[x-size*(.38-i*.04),y-size*.25+dy],[x+size*(.38-i*.04),y-size*.25+dy]],ctx.fillStyle)}
}
function drawWorldObjects(){
 let first=Math.floor(state.distance/22),last=Math.ceil((state.distance+MAX_DIST)/22);
 for(let i=last;i>=first;i--){
  let z=i*22,a=project(z);if(z<=state.distance||z>=state.distance+MAX_DIST)continue;
  let size=clamp(6+82*a.p*a.p,3,75),side=i%2===0?1:-1;
  let xx=a.x+side*(a.half+22+70*a.p);
  tree(xx,a.y+4,size,i);
  if(i%3===0){let x2=a.x-side*(a.half+21+76*a.p);tree(x2,a.y+4,size*.77,i+2)}
 }
}
function roundRect(x,y,w,h,r,fill){ctx.fillStyle=fill;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill()}
function drawCar(){
 const x=W*.5,cy=H*.79, tilt=clamp(state.steer/26,-.65,.65);
 ctx.save();ctx.translate(x,cy);ctx.rotate(tilt*.045);
 // Ground shadow and rotating wheel hints make the vehicle legible as an object in motion.
 ctx.fillStyle='#061e29a1';ctx.beginPath();ctx.ellipse(0,76,114,23,0,0,2*Math.PI);ctx.fill();
 roundRect(-102,-4,26,106,10,'#152732');roundRect(76,-4,26,106,10,'#152732');
 const body=ctx.createLinearGradient(-95,0,95,0);body.addColorStop(0,'#0b5e91');body.addColorStop(.34,'#1ea6ec');body.addColorStop(.69,'#1686bd');body.addColorStop(1,'#064b7f');
 poly([[-83,67],[-89,19],[-73,-32],[-56,-58],[57,-58],[74,-32],[89,19],[83,67],[60,91],[-59,91]],body);
 poly([[-56,-52],[56,-52],[67,-16],[-67,-16]],'#083350');
 poly([[-48,-46],[49,-46],[57,-20],[-56,-20]],'#295c78');
 line([[-50,-43],[50,-43]],'#669dbc',2);
 roundRect(-65,-12,130,15,6,'#115a86');
 ctx.fillStyle='#c5eaf3a0';ctx.fillRect(-40,27,80,5);
 roundRect(-80,67,160,26,8,'#113a58');
 roundRect(-77,66,39,15,5,'#ff574f');roundRect(38,66,39,15,5,'#ff574f');
 roundRect(-69,68,25,7,3,'#ffb48c');roundRect(44,68,25,7,3,'#ffb48c');
 roundRect(-20,75,40,11,3,'#d7ecf4');
 roundRect(-53,-60,107,9,5,'#1a8bc9');
 ctx.restore();
}
function drawScene(){
 ctx.clearRect(0,0,W,H);drawBackdrop();drawRoad();drawWorldObjects();drawCar();
 // Transparent readability vignette at lower corners, inside canvas only.
 let v=ctx.createLinearGradient(0,H-135,0,H);v.addColorStop(0,'#071d2a00');v.addColorStop(1,'#051b2a7a');ctx.fillStyle=v;ctx.fillRect(0,H-135,W,135);
}
function updateSimulation(dt){
 if(!state.running)return;
 dt=clamp(dt,0,.05);
 const speedMeters=state.speed/3.6;
 state.distance+=speedMeters*dt;
 if(state.auto){
   // Proportional pursuit of route centre; visually plausible but non-ML.
   let currentTarget=centerAt(state.distance+11);
   let desiredV=clamp((currentTarget-state.lateral)*.78,-3.6,3.6);
   let deltaX=desiredV*dt;
   state.lateral+=deltaX;
   let desiredSteer=clamp(Math.atan2(desiredV,speedMeters)*180/Math.PI*2.4,-22,22);
   state.steer=lerp(state.steer,desiredSteer,Math.min(1,dt*4.5));
 } else {
   const target=(state.heldRight?19:0)-(state.heldLeft?19:0);
   state.steer=lerp(state.steer,target,Math.min(1,dt*6));
   state.lateral+=state.steer/19 *speedMeters*.24*dt;
   state.lateral=clamp(state.lateral,centerAt(state.distance)-6.3,centerAt(state.distance)+6.3);
 }
}
function steerText(angle){return angle>=0?`+${angle.toFixed(1)}`:angle.toFixed(1)}
function updateUI(){
 $('steerAngle').textContent=steerText(state.steer);
 $('steeringWheel').style.transform=`rotate(${state.steer*5}deg)`;
 $('distance').innerHTML=`${(state.distance/1000).toFixed(2)} <small>km</small>`;
 $('velocity').innerHTML=`${state.speed} <small>km/h</small>`;
 $('speedFloat').innerHTML=`${state.speed} <span>km/h</span>`;
 const dir=Math.abs(state.steer)<1?'Following road centre':state.steer>0?'Steering right':'Steering left';
 $('steerDirection').textContent=state.mode==='sim'?dir:'Recorded steering estimate';
 $('directionArrow').textContent=Math.abs(state.steer)<1?'↑':state.steer>0?'↗':'↖';
}
function frame(ts){
 const dt=state.lastTs?(ts-state.lastTs)/1000:0;state.lastTs=ts;
 if(state.mode==='sim'){updateSimulation(dt);drawScene();updateUI()}
 state.raf=requestAnimationFrame(frame);
}
function selectMode(mode){
 state.mode=mode;
 const sim=mode==='sim';
 road.hidden=!sim;video.hidden=sim;
 $('simulationControls').hidden=!sim;$('replayControls').hidden=sim;
 $('btnSim').classList.toggle('active',sim);$('btnVideo').classList.toggle('active',!sim);
 $('btnSim').setAttribute('aria-pressed',String(sim));$('btnVideo').setAttribute('aria-pressed',String(!sim));
 $('camLabel').innerHTML=sim?'<span class="cam-dot"></span> CHASE CAMERA <span class="camera-divider">/</span> SIMULATION':'<span class="cam-dot"></span> PROCESSED VIDEO <span class="camera-divider">/</span> OPENCV';
 $('frameLabel').textContent=sim?'TRACK 01 · '+$('trackSelect').selectedOptions[0].text.toUpperCase():'160 PROCESSED FRAMES';
 $('liveBadge').innerHTML=sim?'<span class="status-dot"></span> '+(state.running?'RUNNING':'PAUSED'):'<span class="status-dot"></span> REPLAY';
 $('videoNotice').hidden=sim;$('speedFloat').hidden=!sim;
 $('angleLabel').textContent=sim?'SIMULATED STEERING ANGLE':'MEASURED LANE-GEOMETRY ANGLE';
 $('disclaimerTitle').textContent=sim?'Simulation only — not AI driving.':'Replay only — not CNN inference.';
 $('disclaimerBody').textContent=sim?'The animated car uses a simple route-following controller. It does not execute a trained neural network.':'Angles are from the OpenCV baseline applied to synthetic footage. A trained CNN is not loaded in this static site.';
 $('autopilot').disabled=!sim;
 if(sim){video.pause();$('driveModeInfo').textContent=state.auto?'Visual route-following controller':'Manual arrow-key steering';updateUI()}
 else{video.currentTime=0;video.play().catch(()=>{});updateFromVideo()}
}
// The recording's real computed angle trace is generated by Python and stored alongside demo assets.
let replayAngles=[];
function updateFromVideo(){
 if(state.mode!=='video')return;
 const frameNumber=clamp(Math.floor(video.currentTime*20),0,159),angle=replayAngles[frameNumber]??0;
 $('seek').value=String(frameNumber);$('videoTime').textContent=`0:${String(Math.floor(video.currentTime)).padStart(2,'0')} / 0:08`;
 $('steerAngle').textContent=steerText(angle);$('steeringWheel').style.transform=`rotate(${angle*5}deg)`;
 $('steerDirection').textContent=Math.abs(angle)<.6?'Road centre aligned':angle>0?'Lane centre right':'Lane centre left';
 $('directionArrow').textContent=Math.abs(angle)<.6?'↑':angle>0?'↗':'↖';
 $('distance').innerHTML=`${(video.currentTime).toFixed(1)} <small>s</small>`;
 $('velocity').innerHTML=`${frameNumber+1} <small>/ 160 frames</small>`;
 $('videoPlay').textContent=video.paused?'▶ Play replay':'Ⅱ Pause replay';
}
async function loadReplayAngles(){
 const embedded=document.getElementById('angle-trace');
 if(embedded){try{replayAngles=JSON.parse(embedded.textContent);if(replayAngles.length===160)return;}catch(err){console.warn('Invalid embedded trace:',err)}}
 try{const response=await fetch('angles.json');if(!response.ok)throw new Error('No trace');const json=await response.json();if(!Array.isArray(json)||json.length!==160)throw new Error('Unexpected trace');replayAngles=json}
 catch(err){replayAngles=[];console.warn('Replay angle trace unavailable:',err)}
}
$('playPause').addEventListener('click',()=>{state.running=!state.running;$('playText').textContent=state.running?'Pause drive':'Resume drive';$('playGlyph').textContent=state.running?'Ⅱ':'▶';$('liveBadge').innerHTML='<span class="status-dot"></span> '+(state.running?'RUNNING':'PAUSED')});
$('restart').addEventListener('click',()=>{state.distance=0;state.lateral=centerAt(0);state.steer=0;updateUI();drawScene()});
$('trackSelect').addEventListener('change',e=>{state.route=e.target.value;state.distance=0;state.lateral=centerAt(0);state.steer=0;$('frameLabel').textContent='TRACK 01 · '+e.target.selectedOptions[0].text.toUpperCase()});
$('speedSlider').addEventListener('input',e=>{state.speed=+e.target.value;$('speedValue').textContent=state.speed+' km/h';updateUI()});
$('autopilot').addEventListener('change',e=>{state.auto=e.target.checked;$('driveModeInfo').textContent=state.auto?'Visual route-following controller':'Manual arrow-key steering'});
$('btnSim').addEventListener('click',()=>selectMode('sim'));
$('btnVideo').addEventListener('click',()=>selectMode('video'));
$('openReplay').addEventListener('click',()=>{selectMode('video');$('demo').scrollIntoView({behavior:'smooth',block:'start'})});
$('videoPlay').addEventListener('click',()=>{video.paused?video.play().catch(()=>{}):video.pause();updateFromVideo()});
$('seek').addEventListener('input',e=>{video.pause();video.currentTime=+e.target.value/20;updateFromVideo()});
video.addEventListener('timeupdate',updateFromVideo);video.addEventListener('play',updateFromVideo);video.addEventListener('pause',updateFromVideo);
window.addEventListener('keydown',e=>{if(state.mode!=='sim'||(['SELECT','TEXTAREA'].includes(document.activeElement.tagName)||(document.activeElement.tagName==='INPUT'&&document.activeElement.type!=='checkbox')))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){state.heldLeft=e.key==='ArrowLeft'?true:state.heldLeft;state.heldRight=e.key==='ArrowRight'?true:state.heldRight;e.preventDefault()}});
window.addEventListener('keyup',e=>{if(e.key==='ArrowLeft')state.heldLeft=false;if(e.key==='ArrowRight')state.heldRight=false});
window.addEventListener('blur',()=>{state.heldLeft=false;state.heldRight=false});
state.lateral=centerAt(0);loadReplayAngles();updateUI();state.raf=requestAnimationFrame(frame);

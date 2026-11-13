import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const config = await fetch('/api/config').then(r => r.json());
const container = $('scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef2f3);
const camera = new THREE.PerspectiveCamera(36, 1, .01, 100);
const renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 1.7;
controls.maxDistance = 9;
controls.maxPolarAngle = Math.PI*.91;
scene.add(new THREE.HemisphereLight(0xffffff, 0xa2b5bb, 2.6));
const light = new THREE.DirectionalLight(0xffffff, 3);
light.position.set(3, 5, 3);
scene.add(light);
const grid = new THREE.GridHelper(12, 60, 0xb1c1c5, 0xd7e0e2);
scene.add(grid);
const axes = new THREE.AxesHelper(.35);
axes.position.set(-1.4, .01, .8);
scene.add(axes);
const geo = new THREE.BufferGeometry();
geo.setIndex(config.faces);
const body = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:0x65a9a0, roughness:.76, metalness:.02, side:THREE.DoubleSide}));
scene.add(body);
const bonesGeo = new THREE.BufferGeometry();
const segments = config.parents.map((p,j)=>p>=0 && p!==j ? [p,j] : null).filter(Boolean);
bonesGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segments.length*6),3));
const bones = new THREE.LineSegments(bonesGeo,new THREE.LineBasicMaterial({color:0x303c53,depthTest:false}));
bones.renderOrder=3;
bones.visible=false;
scene.add(bones);
let frameSeq=-1, playing=true, initial=true, latestState=null, busy=false, frameBusy=false;
let frameLoads=0;
window.__fgpPreview={renderer,scene,camera,get frameLoads(){return frameLoads;}};

function home(){camera.position.set(3,1.7,4.5);controls.target.set(0,.1,0);controls.update();}
home();
const observer=new ResizeObserver(()=>{const w=container.clientWidth,h=container.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();});
observer.observe(container);
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
function icons(){window.lucide.createIcons();}
icons();
async function command(data){
  const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Preview-Token':config.token},body:JSON.stringify(data)});
  const result=await response.json();
  if(!response.ok) throw new Error(result.error);
}
$('home').onclick=home;
for(const [id,obj] of [['body',body],['bones',bones]]) $(''+id).onclick=()=>{obj.visible=!obj.visible;$(id).setAttribute('aria-pressed',String(obj.visible));};
$('play').onclick=async()=>{try{await command({action:'play',playing:!playing});}catch(e){$('error').textContent=e.message;}};
$('seek').onchange=async e=>{try{await command({action:'seek',frame:Number(e.target.value)});}catch(e){$('error').textContent=e.message;}};
$('calibrate').onclick=async()=>{try{await command({action:'calibrate'});}catch(e){$('error').textContent=e.message;}};
async function mesh(){
  if(frameBusy)return;
  frameBusy=true;
  try{
    const response=await fetch('/api/frame');
    if(!response.ok)throw new Error('人体网格读取失败');
    const data=new Float32Array(await response.arrayBuffer());
    const vertices=data.slice(72);
    if(!geo.getAttribute('position'))geo.setAttribute('position',new THREE.BufferAttribute(vertices,3));
    else{geo.attributes.position.array.set(vertices);geo.attributes.position.needsUpdate=true;}
    geo.computeVertexNormals();geo.computeBoundingSphere();
    const pos=bonesGeo.attributes.position;
    segments.forEach(([p,j],i)=>{pos.array.set(data.slice(p*3,p*3+3),i*6);pos.array.set(data.slice(j*3,j*3+3),i*6+3);});
    pos.needsUpdate=true;bonesGeo.computeBoundingSphere();
    if(initial){
      let minY=Infinity;for(let i=1;i<vertices.length;i+=3)minY=Math.min(minY,vertices[i]);
      grid.position.y=minY;axes.position.y=minY+.01;
      controls.target.set(data[0],data[1]+.1,data[2]);initial=false;
    }
    frameLoads++;
  }finally{frameBusy=false;}
}
const states={idle:'等待',scanning:'扫描中',discovered:'已发现',connecting:'连接中',subscribed:'已订阅',receiving:'接收中',retry_wait:'等待重连',error:'连接错误',not_seen:'扫描未见',disconnected:'已断开'};
const fmt=(value,digits=1)=>Number.isFinite(value)?value.toFixed(digits):'—';
function draw(state){
  latestState=state;
  const replay=state.mode==='replay';
  $('replay-controls').hidden=!replay;
  for(const id of ['live-controls','calibration-controls','sensor-section'])$(id).hidden=replay;
  $('source-badge').textContent=replay?'历史录制':'实时 BLE';
  $('source-badge').className='badge'+(replay?'':' live');
  $('scene-source').textContent=replay?'CLOTHO · GlobalPose · 历史录制':'SmartWear · 传感器实时数据';
  $('source-name').textContent=state.source_name;
  $('reconstruction').textContent=replay?'最新 FGP · 录制结果':({not_enabled:'未启用',waiting_calibration:'等待标定',history_warmup:'历史帧积累中',stale_hold:'数据过期 · 暂存',receiving:state.smpl_realtime_valid?'最新 FGP · 实时':'最新 FGP · 数据过期',error:'重建错误'})[state.reconstruction]||state.reconstruction;
  $('clock').textContent=replay?'录制时间轴':'接收端单调时钟';
  $('pose-label').textContent=replay?'历史录制 · 非当前穿戴者':state.smpl_realtime_valid?'实时 CLOTHO · GlobalPose':state.frame_seq>0?'SMPL 暂存 · 非实时姿态':'静态模板 · 非人体推理';
  $('smpl-age').textContent=state.smpl_age_ms==null?'—':`${fmt(state.smpl_age_ms,0)} ms`;
  $('inference-ms').textContent=state.smpl_inference_ms==null?'—':`${fmt(state.smpl_inference_ms)} ms`;
  $('so3-error').textContent=state.smpl_so3_error==null?'—':state.smpl_so3_error.toExponential(2);
  $('error').textContent=state.error||'';
  $('footer-state').textContent=replay?'录制原始时间 · 15 Hz 预览':'11 个物理 IMU · 10 个模型输入';
  $('connection-dot').className='dot'+(replay||Object.values(state.devices).every(d=>d.fresh)?' ok':'');
  if(replay){
    if(playing!==state.playing){$('play').innerHTML=`<i data-lucide="${state.playing?'pause':'play'}"></i>`;icons();}
    playing=state.playing;$('play').title=playing?'暂停':'播放';$('play').setAttribute('aria-label',$('play').title);
    $('seek').max=state.frame_count-1;
    if(document.activeElement!==$('seek'))$('seek').value=state.frame;
    $('position').textContent=`${state.frame+1} / ${state.frame_count}`;
  }else{
    $('devices').replaceChildren();$('imu-rows').replaceChildren();
    for(const [role,d] of Object.entries(state.devices)){
      const node=document.createElement('div');node.className='device';
      node.innerHTML=`<div class="device-head"><b>${role==='upper'?'上衣':'裤子'}</b><span>${states[d.state]||d.state}</span></div><div class="device-info"><span>${d.address}</span><span>${fmt(d.hz)} Hz · ${fmt(d.age_ms,0)} ms</span></div>`;
      if(d.error){const error=document.createElement('div');error.className='cal-state';error.textContent=d.error;node.append(error);}
      $('devices').append(node);
      (d.imu_values||[]).forEach((v,i)=>{const row=document.createElement('tr');row.innerHTML=`<td>${role==='upper'?'上衣':'裤子'} ${i+1}</td><td>${v.slice(0,3).map(x=>fmt(x)).join(' / ')}</td><td>${fmt(Math.hypot(...v.slice(3)),2)}</td>`;$('imu-rows').append(row);});
    }
    const c=state.calibration;
    $('calibrate').disabled=!Object.values(state.devices).every(d=>d.fresh)||c.state==='collecting';
    $('cal-state').textContent=({not_captured:'尚未采集',collecting:`采集中 · 剩余 ${fmt(c.remaining_s)} 秒`,static_passed:'静稳检查通过 · T-pose 姿势待人工确认',unstable:'静稳检查未通过',epoch_changed:'设备计数已重置 · 需要重新标定',error:'采集失败'})[c.state]||c.state;
    $('cal-samples').textContent=c.samples??'—';
    $('cal-jitter').textContent=c.orientation_jitter_p95_deg?`${fmt(Math.max(...c.orientation_jitter_p95_deg))}°`:'—';
    $('cal-acc').textContent=c.acceleration_std_norm_m_s2?`${fmt(Math.max(...c.acceleration_std_norm_m_s2),2)} m/s²`:'—';
  }
}
async function poll(){
  if(busy)return;busy=true;
  try{
    const response=await fetch('/api/state');if(!response.ok)throw new Error('服务暂未响应');
    const state=await response.json();draw(state);
    if(frameSeq!==state.frame_seq){await mesh();frameSeq=state.frame_seq;}
  }catch(e){$('error').textContent=e.message;$('connection-dot').className='dot';$('footer-state').textContent='预览连接等待中';}
  finally{busy=false;}
}
await poll();setInterval(poll,67);

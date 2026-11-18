const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const runtimeRequire = createRequire(path.join(process.env.FGP_NODE_MODULES, '_resolver.cjs'));
const {chromium} = runtimeRequire('playwright');
const {PNG} = runtimeRequire('pngjs');

(async()=>{
  const url = process.argv[2] || 'http://127.0.0.1:8768';
  const output = process.argv[3];
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true});
  const errors=[];
  const results=[];
  try{
    for(const viewport of [{width:1366,height:900},{width:390,height:844}]){
      const page=await browser.newPage({viewport});
      page.on('pageerror',error=>errors.push(error.message));
      await page.goto(url);
      await page.waitForFunction(()=>window.__fgpPreview?.frameLoads>1);
      const canvas=page.locator('canvas');
      const timingBefore=await page.evaluate(async()=>fetch('/api/state').then(r=>r.json()));
      const first=PNG.sync.read(await canvas.screenshot());
      await page.waitForTimeout(700);
      const second=PNG.sync.read(await canvas.screenshot());
      const timingAfter=await page.evaluate(async()=>fetch('/api/state').then(r=>r.json()));
      const sourceDelta=(timingAfter.replay_playhead_s-timingBefore.replay_playhead_s+timingAfter.replay_duration_s)%timingAfter.replay_duration_s;
      assert(sourceDelta>.5 && sourceDelta<1.4,`recording time did not follow wall time: ${sourceDelta}`);
      let colored=0,changed=0;
      for(let i=0;i<second.data.length;i+=4){
        if(Math.abs(second.data[i]-238)+Math.abs(second.data[i+1]-242)+Math.abs(second.data[i+2]-243)>70)colored++;
        if(Math.abs(second.data[i]-first.data[i])+Math.abs(second.data[i+1]-first.data[i+1])+Math.abs(second.data[i+2]-first.data[i+2])>40)changed++;
      }
      assert(colored>1000,`canvas blank: ${colored}`);
      assert(changed>100,`canvas not moving: ${changed}`);
      await page.getByRole('button',{name:'暂停',exact:true}).click();
      await page.waitForTimeout(300);
      const before=await page.evaluate(async()=>fetch('/api/state').then(r=>r.json()));
      await page.waitForTimeout(400);
      const after=await page.evaluate(async()=>fetch('/api/state').then(r=>r.json()));
      assert.equal(before.frame,after.frame);
      await page.locator('#seek').fill('20');
      await page.locator('#seek').dispatchEvent('change');
      await page.waitForTimeout(250);
      assert.equal((await page.evaluate(async()=>fetch('/api/state').then(r=>r.json()))).frame,20);
      await page.getByRole('button',{name:'关节骨架',exact:true}).click();
      assert.equal(await page.locator('#bones').getAttribute('aria-pressed'),'true');
      const unauthorized=await page.evaluate(async()=>fetch('/api/command',{method:'POST',body:'{}'}).then(r=>r.status));
      assert.equal(unauthorized,403);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      assert(!overflow,'horizontal overflow');
      await page.screenshot({path:path.join(output,`preview_${viewport.width}.png`),fullPage:true});
      results.push({viewport,colored_pixels:colored,changed_pixels:changed,source_time_delta_s:sourceDelta,paused_frame:before.frame,seek_frame:20,horizontal_overflow:overflow});
      await page.getByRole('button',{name:'播放',exact:true}).click();
      await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'web_checks.json'),JSON.stringify({status:'PASSED',results,errors},null,2)+'\n');
    console.log(JSON.stringify({status:'PASSED',results,errors}));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

"""Exercise the offline coin selector in a real browser without dumping report data."""
import argparse
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import tempfile

from app.backtesting.html_report import read_report, CHARTS


def check(path,rows_per_coin):
    payload=read_report(path)
    coins=payload['spec']['symbols']
    if len(payload['results']['totals'])!=len(coins)*rows_per_coin:
        raise ValueError('Unexpected comparison matrix')
    script='''<script>
(async()=>{
 const delay=ms=>new Promise(r=>setTimeout(r,ms));
 for(let i=0;i<150&&!Array.from(document.images).every(x=>x.complete);i++)await delay(100);
 let result=[];
 for(const value of ['',...COINS]){
  document.getElementById('coin').value=value;filterRows();await delay(50);
  let rows=Array.from(document.querySelectorAll('.comparison tbody tr')).filter(x=>!x.hidden);
  let pictures=Array.from(document.querySelectorAll('.plots img')).filter(x=>x.getClientRects().length>0);
  result.push({coin:value||'ALL',rows:rows.length,images:pictures.length,
               all_loaded:pictures.every(x=>x.complete&&x.naturalWidth>0),
               positive_width:pictures.every(x=>x.getBoundingClientRect().width>0)});
 }
 const node=document.createElement('pre');node.id='browser-qa-results';node.textContent=JSON.stringify(result);document.body.append(node);
})();</script>'''.replace('COINS',json.dumps(coins))
    with tempfile.TemporaryDirectory(prefix='hjb-browser-qa-') as directory:
        folder=Path(directory)
        source=folder/'report.html'
        source.write_text(path.read_text().replace('</body>',script+'</body>'))
        with (folder/'dom.html').open('w') as out,(folder/'chrome.log').open('w') as err:
            subprocess.run(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                            '--headless','--no-sandbox','--disable-gpu',
                            '--user-data-dir='+str(folder/'chrome-profile'),
                            '--virtual-time-budget=20000','--dump-dom',source.as_uri()],
                           stdout=out,stderr=err,check=True,timeout=90)
        dom=(folder/'dom.html').read_text()
        match=re.search(r'<pre id="browser-qa-results">([^<]*)</pre>',dom)
        if match is None:raise ValueError('Browser QA did not finish')
        results=json.loads(unescape(match.group(1)))
    for row in results:
        expected_rows=rows_per_coin*(len(coins) if row['coin']=='ALL' else 1)
        fee_groups=len(payload['spec'].get('fee_scenarios',[])) if payload.get('fee_split_charts') else 1
        expected_images=len(CHARTS)*max(1,fee_groups)*(len(coins) if row['coin']=='ALL' else 1)
        if row['rows']!=expected_rows or row['images']!=expected_images or not(row['all_loaded'] and row['positive_width']):
            raise ValueError('Browser QA failed: '+json.dumps(row))
    if len(results)!=len(coins)+1:raise ValueError('Incomplete browser selector check')
    print(json.dumps(dict(status='passed',rows=results),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    parser.add_argument('--rows-per-coin',type=int,required=True)
    a=parser.parse_args()
    check(a.report,a.rows_per_coin)

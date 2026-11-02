import os
import shutil
"""Transaction import, schema migration and cross-tab safety. Local server on 8765."""
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:8765/'
fixture=Path('tests/fixtures/trading212-history.csv').read_text()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_EXECUTABLE') or shutil.which('chromium'),headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
    page=context.new_page();errors=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    context.on('request',lambda r:requests.append((r.method,r.url)))
    # Seed the storage format from the first release without loading the new application.
    page.goto(BASE+'options-skew.html')
    page.evaluate('''() => new Promise((resolve,reject)=>{const q=indexedDB.open('stocks-research-private',1);q.onupgradeneeded=()=>q.result.createObjectStore('state');q.onsuccess=()=>{const d=q.result,t=d.transaction('state','readwrite');t.objectStore('state').put({schema:1,portfolios:[{id:'old-private',name:'Existing private portfolio',currency:'GBP',cash:0,positions:[{symbol:'OLD',quantity:1,currency:'GBP',cost:100,price:100}]}],watchlist:[],journal:[],imports:[],preferences:{currency:'GBP',hidden:[]}},'main');t.oncomplete=()=>{d.close();resolve();};};q.onerror=()=>reject(q.error);})''')
    page.goto(BASE+'#portfolio');page.wait_for_selector('#portfolio-select')
    assert page.locator('#portfolio-select').input_value()=='old-private'
    def stored():
        return page.evaluate("async()=>{const m=await import('./assets/portfolio.mjs');const d=await m.openStore();const s=await m.readStore(d);d.close();return s;}")
    def upload(text,name):
        page.get_by_role('button',name='Import CSV',exact=False).click()
        page.locator('#csv-file').set_input_files({'name':name,'mimeType':'text/csv','buffer':text.encode()})
        page.get_by_role('button',name='Review reconciliation',exact=True).click()
        page.wait_for_selector('#history-destination')
    def review(destination=''):
        page.locator('#history-destination').select_option(destination)
        page.locator('#history-complete').check();page.locator('#history-net').check()
        page.get_by_role('button',name='Reconcile transactions',exact=True).click()
    upload(fixture,'history.csv')
    page.get_by_role('button',name='Reconcile transactions',exact=True).click()
    assert 'Confirm' in page.locator('#history-error').inner_text()
    review();page.locator('#history-name').fill('Book history')
    assert page.get_by_role('button',name='Save reconciled portfolio').is_enabled()
    page.get_by_role('button',name='Save reconciled portfolio').click()
    page.wait_for_function("!document.querySelector('dialog').open")
    s=stored();assert s['schema']==2 and len(s['portfolios'])==2
    ledger=next(x for x in s['portfolios'] if x.get('ledger'));lid=ledger['id']
    assert ledger['cash']==925.5 and ledger['positions'][0]['basisBase']==48.5
    assert len(ledger['ledger']['transactions'])==8
    assert s['portfolios'][0]['positions'][0]['symbol']=='OLD'
    page.get_by_role('button',name='Edit',exact=True).click()
    assert page.locator('#position-form [name=quantity]').get_attribute('readonly') is not None
    page.locator('#position-form [name=price]').fill('140')
    page.locator('#position-form [name=asOf]').fill('2026-01-08')
    page.get_by_role('button',name='Save position',exact=True).click()
    page.wait_for_function("!document.querySelector('dialog').open")
    page.reload();page.wait_for_selector('#portfolio-select');assert page.locator('#portfolio-select').input_value()==lid
    more=fixture+'Deposit,2026-01-09 09:00:00,,,,,cash-3,,,,,,,50,GBP,,\n'
    upload(more,'overlap.csv');review(lid)
    assert '1 new records' in page.locator('dialog').inner_text()
    assert '8 exact broker-ID duplicates skipped' in page.locator('dialog').inner_text()
    page.get_by_role('button',name='Append reconciled records').click()
    page.wait_for_function("!document.querySelector('dialog').open")
    s=stored();ledger=next(x for x in s['portfolios'] if x['id']==lid)
    assert ledger['cash']==975.5 and len(ledger['ledger']['transactions'])==9
    assert ledger['positions'][0]['price']==140 and ledger['positions'][0]['basisBase']==48.5
    before=s
    upload(fixture.replace('1000,GBP','1001,GBP'),'conflict.csv');review(lid)
    assert page.get_by_role('button',name='Append reconciled records').is_disabled()
    assert 'conflicts' in page.locator('dialog').inner_text()
    page.get_by_role('button',name='Close dialog').click();assert stored()==before
    page.get_by_role('button',name='Review transaction ledger',exact=True).click()
    assert '69.50' in page.locator('dialog').inner_text()
    page.get_by_role('button',name='Close dialog').click()
    # Backups preserve ledger validation and use the current local revision guard.
    import json
    snapshot=stored()
    page.get_by_role('button',name='Restore backup',exact=True).click()
    broken=json.loads(json.dumps(snapshot));next(x for x in broken['portfolios'] if x.get('ledger'))['cash']+=1
    page.locator('#backup-file').set_input_files({'name':'bad.json','mimeType':'application/json','buffer':json.dumps(broken).encode()})
    page.wait_for_function("document.querySelector('#backup-error').textContent.includes('reconcile')")
    assert stored()==snapshot
    page.get_by_role('button',name='Close dialog').click()
    page.get_by_role('button',name='Restore backup',exact=True).click()
    page.locator('#backup-file').set_input_files({'name':'backup.json','mimeType':'application/json','buffer':json.dumps(snapshot).encode()})
    page.get_by_role('button',name='Replace local workspace',exact=True).click()
    page.wait_for_function("!document.querySelector('dialog').open")
    assert stored()['portfolios']==snapshot['portfolios']
    assert stored()['revision']==snapshot['revision']+1
    # A stale tab must not overwrite another tab's newly saved data.
    stale=context.new_page();stale.goto(BASE+'#portfolio');stale.wait_for_selector('#portfolio-select')
    page.get_by_role('button',name='Find a stock').click();page.locator('#symbol-search').fill('AAPL')
    page.locator('#search-results button').first.click();page.get_by_role('button',name='Add to watchlist',exact=True).click()
    page.wait_for_function("document.querySelector('#toast').textContent.includes('updated')")
    stale.get_by_role('button',name='New portfolio',exact=False).click();stale.locator('#new-form [name=name]').fill('Must not overwrite')
    stale.get_by_role('button',name='Create in GBP').click()
    stale.wait_for_function("document.querySelector('#toast').textContent.includes('another tab')")
    assert len(stored()['portfolios'])==2 and 'AAPL' in stored()['watchlist']
    stale.close();page.get_by_role('button',name='Close dialog').click()
    for width in [1440,768,390]:
        page.set_viewport_size({'width':width,'height':900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.get_by_role('button',name='Remove all local data',exact=True).click()
    page.get_by_role('button',name='Remove everything',exact=True).click()
    page.wait_for_function("!document.querySelector('dialog').open")
    cleared=stored();assert cleared['portfolios']==[] and cleared['watchlist']==[] and cleared['imports']==[]
    # An unreadable workspace must reject writes, not silently replace its data.
    isolated=browser.new_context();broken_page=isolated.new_page()
    broken_page.on('pageerror',lambda e:errors.append(str(e)))
    broken_page.goto(BASE+'options-skew.html')
    broken_page.evaluate("""()=>new Promise(resolve=>{const q=indexedDB.open('stocks-research-private',2);q.onupgradeneeded=()=>q.result.createObjectStore('state');q.onsuccess=()=>{const d=q.result,t=d.transaction('state','readwrite');t.objectStore('state').put({schema:99,sentinel:'preserve'},'main');t.oncomplete=()=>{d.close();resolve();};};})""")
    broken_page.goto(BASE+'#portfolio');broken_page.wait_for_function("document.querySelector('#main').textContent.includes('Saving is disabled')")
    broken_page.get_by_role('button',name='New portfolio',exact=False).click();broken_page.locator('#new-form [name=name]').fill('Cannot overwrite')
    broken_page.get_by_role('button',name='Create in GBP').click()
    broken_page.wait_for_function("document.querySelector('#toast').textContent.includes('Storage unavailable')")
    raw=broken_page.evaluate("""()=>new Promise(resolve=>{const q=indexedDB.open('stocks-research-private',2);q.onsuccess=()=>{const d=q.result,r=d.transaction('state').objectStore('state').get('main');r.onsuccess=()=>{d.close();resolve(r.result);};};})""")
    assert raw=={'schema':99,'sentinel':'preserve'};isolated.close()
    assert not errors,errors
    assert all(method=='GET' and url.startswith(BASE) for method,url in requests),requests
    print('PASS: schema 1 upgrade; FIFO history import; quote-only edit; overlapping imports; conflict atomicity; active portfolio persistence; cross-tab protection; validated backup restore; reset; corrupt-storage preservation; no holdings transmission; 3 widths')
    browser.close()

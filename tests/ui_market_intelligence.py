"""Public-data valuation, risk modelling, staleness and local privacy regression."""
import json
import os
import shutil
from datetime import datetime,timedelta,timezone
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:8765/'

def public_fixture():
    end=datetime(2026,11,3,tzinfo=timezone.utc);days=[];d=end
    while len(days)<90:
        if d.weekday()<5:days.append(d.strftime('%Y-%m-%d'))
        d-=timedelta(days=1)
    days.reverse()
    def bars(factor):
        price=100.0;rows=[]
        for i,day in enumerate(days):
            price*=1.01 if i%2==0 else .995
            rows.append({'date':day,'open':price*factor,'high':(price+2)*factor,'low':(price-2)*factor,'close':price*factor,'adjusted_close':price*factor,'volume':1000})
        return rows
    def instrument(symbol,factor):
        return {'symbol':symbol,'currency':'USD','status':'ok','exchange':'NMS','instrument_type':'EQUITY','last_session':days[-1],'history':bars(factor)}
    extra=instrument('LOCAL.L',1);extra['exchange']='LSE'
    return {'schema_version':1,'generated_at':'2026-11-04T22:30:00Z','coverage':{'available':3,'requested':3},'instruments':{**{s:instrument(s,f) for s,f in [('AAPL',1),('MSFT',2),('SPY',1)]},'LOCAL.L':extra},'fx':{'GBP':{'symbol':'GBPUSD=X','currency':'USD','unit_currency':'GBP','status':'ok','history':[{'date':day,'usd_per_unit':1.25} for day in days]}}}

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_EXECUTABLE') or shutil.which('chromium'),headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1000})
    fixture=public_fixture();outage=False;requests=[];errors=[]
    def route_market(route):
        if outage:route.fulfill(status=503,body='Unavailable')
        else:route.fulfill(status=200,content_type='application/json',body=json.dumps(fixture))
    context.route('**/data/portfolio-market.json',route_market)
    context.on('request',lambda r:requests.append((r.method,r.url)))
    page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.clock.set_fixed_time(datetime(2026,11,4,23,tzinfo=timezone.utc))
    page.goto(BASE+'#portfolio');page.wait_for_selector('h1')
    page.get_by_role('button',name='Import CSV',exact=False).click()
    csv=b'Ticker,Quantity,Currency,Average price,Current price\nAAPL,2,USD,100,10\nMSFT,1,USD,200,20\n'
    page.locator('#csv-file').set_input_files({'name':'positions.csv','mimeType':'text/csv','buffer':csv})
    page.get_by_role('button',name='Review reconciliation',exact=True).click()
    page.get_by_role('button',name='Save new portfolio',exact=True).click()
    page.wait_for_selector('#quote-mode')
    assert page.locator('#quote-mode').input_value()=='manual'
    def stored():
        return page.evaluate("async()=>{const m=await import('./assets/portfolio.mjs'),db=await m.openStore();const s=await m.readStore(db);db.close();return s;}")
    before=stored()['portfolios'][0]['positions']
    page.locator('#quote-mode').select_option('published')
    page.wait_for_function("document.querySelector('#main').textContent.includes('Model annualised volatility')")
    assert stored()['portfolios'][0]['positions']==before
    assert page.get_by_text('Historical base-currency cost is unavailable',exact=False).count()==1
    model=page.locator('section').filter(has=page.get_by_role('heading',name='Historical allocation model · not actual performance',exact=True)).first
    assert '89 aligned observations' in model.inner_text()
    assert 'High correlation' in model.inner_text()
    price=fixture['instruments']['AAPL']['history'][-1]['close']
    expected=round((2*price+2*price)/1.25,2)
    assert page.get_by_text(f'£{expected:,.2f}',exact=True).count()>0
    page.locator('#what-symbol').select_option('0')
    page.get_by_role('button',name='Calculate allocation scenario',exact=True).click()
    assert 'hypothetical purchase' in page.locator('#what-result').inner_text()
    page.reload();page.wait_for_selector('#quote-mode');assert page.locator('#quote-mode').input_value()=='published'
    page.get_by_role('button',name='Find a stock').click();page.locator('#symbol-search').fill('AAPL')
    page.locator('#search-results button').first.click()
    assert page.get_by_role('heading',name='Daily technical context',exact=True).count()==1
    assert 'RSI (14)' in page.locator('dialog').inner_text()
    page.get_by_role('button',name='Close dialog').click()
    page.get_by_role('button',name='Find a stock').click();page.locator('#symbol-search').fill('LOCAL.L')
    page.locator('#search-results button').first.click();assert page.locator('dialog h1').inner_text()=='LOCAL.L'
    page.get_by_role('button',name='Close dialog').click()
    for width in [1440,768,390]:
        page.set_viewport_size({'width':width,'height':900})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    # A new publication timestamp cannot make old quotes fresh.
    fixture['instruments']['AAPL']['history']=[r for r in fixture['instruments']['AAPL']['history'] if r['date']<='2026-10-28']
    fixture['instruments']['AAPL']['last_session']='2026-10-28'
    page.reload();page.wait_for_selector('#quote-mode')
    assert 'older-than-four-day closing price' in page.locator('#main').inner_text()
    assert page.get_by_role('heading',name='Model annualised volatility',exact=True).count()==0
    fixture=public_fixture();fixture['instruments']['AAPL']['currency']='GBP'
    page.reload();page.wait_for_selector('#quote-mode')
    assert 'Quote currency mismatch' in page.locator('#main').inner_text()
    outage=True
    page.reload();page.wait_for_selector('#quote-mode')
    assert 'absent or unavailable' in page.locator('#main').inner_text()
    assert stored()['portfolios'][0]['positions']==before
    page.locator('#quote-mode').select_option('manual')
    page.wait_for_function("document.querySelector('#quote-mode').value==='manual'")
    assert stored()['portfolios'][0]['positions']==before
    assert not errors,errors
    assert all(method=='GET' and url.startswith(BASE) for method,url in requests),requests
    print('PASS: FX valuation; historical cost withheld; risk/correlation model; indicators; stale/mismatched/outage states; no quote mutation or private transmission; 3 widths')
    browser.close()

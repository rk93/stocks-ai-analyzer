"""Run against `python -m http.server 8765 --directory docs` with Playwright installed."""
from playwright.sync_api import sync_playwright
BASE='http://127.0.0.1:8765/'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE);page.wait_for_selector('h1')
    page.get_by_role('button',name='Try demo').click();page.wait_for_selector('#portfolio-select')
    page.reload();page.wait_for_selector('#portfolio-select')
    assert page.locator('#portfolio-select option').count()==1
    page.get_by_role('button',name='Import CSV',exact=False).click()
    csv=b'Ticker,Quantity,Currency,Average price,Current price\nTEST,2,GBP,100,120\n'
    page.locator('#csv-file').set_input_files({'name':'positions.csv','mimeType':'text/csv','buffer':csv})
    page.get_by_role('button',name='Review reconciliation').click()
    page.get_by_role('button',name='Save new portfolio').click()
    page.wait_for_selector('#portfolio-select option:nth-child(2)',state='attached')
    assert page.get_by_text('£240.00',exact=True).count()>0
    page.get_by_role('button',name='Import CSV',exact=False).click()
    page.locator('#csv-file').set_input_files({'name':'duplicate.csv','mimeType':'text/csv','buffer':csv})
    page.wait_for_function("document.querySelector('#import-error').textContent.includes('already imported')")
    page.get_by_role('button',name='Close dialog').click()
    assert page.locator('#portfolio-select option').count()==2
    page.get_by_role('button',name='Import CSV',exact=False).click()
    page.locator('#csv-file').set_input_files({'name':'invalid.csv','mimeType':'text/csv','buffer':b'Ticker,Quantity,Currency,Average price\nBROKEN,-1,GBP,3'})
    page.get_by_role('button',name='Review reconciliation').click()
    assert page.get_by_role('button',name='Save new portfolio').is_disabled()
    page.get_by_role('button',name='Close dialog').click()
    assert page.locator('#portfolio-select option').count()==2
    page.get_by_role('button',name='Find a stock').click();page.locator('#symbol-search').fill('AAPL')
    page.locator('#search-results button').first.click();assert page.locator('dialog h1').inner_text()=='AAPL'
    page.get_by_role('button',name='Compare symbols').click();page.wait_for_selector('#compare-second')
    page.get_by_role('button',name='Close dialog').click()
    for width in [1440,768,390]:
        page.set_viewport_size({'width':width,'height':900})
        for route in ['overview','discover','portfolio','lab','insights']:
            page.goto(BASE+'#'+route);page.wait_for_selector('h1');page.wait_for_timeout(100)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),(width,route)
    page.goto(BASE+'options-skew.html');page.wait_for_selector('#count')
    assert page.get_by_role('link',name='My Portfolio · private').count()==1
    assert not errors,errors
    print('PASS: routes at 3 widths; IndexedDB persistence; import/duplicate/failed-import atomicity; stock comparison; legacy navigation; no browser errors')
    browser.close()

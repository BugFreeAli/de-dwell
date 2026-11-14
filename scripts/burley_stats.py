import csv, statistics as st
rows=list(csv.DictReader(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','docs','burley_tables_transcribed.csv'))))
for r in rows:
    for k in ('dt1045_s','rmse1045_s','dt1315_s','rmse1315_s'): r[k]=float(r[k])
    r['rel1045']=r['rmse1045_s']/r['dt1045_s']
    r['rel1315']=r['rmse1315_s']/r['dt1315_s']
G={'1976 US Standard':['1976 US Standard'],'Climatology':['50% Climatology'],'Ground tab':['Ground Tab'],'GFS 48/24/12 h':['48 Hr GFS','24 Hr GFS','12 Hr GFS'],'GFS 12 h only':['12 Hr GFS'],'Climatology+ground tab':['50% Climatology','Ground Tab'],'All six':list(dict.fromkeys(r['profile'] for r in rows))}
def q(x,p):
    x=sorted(x); i=(len(x)-1)*p; lo=int(i); hi=min(lo+1,len(x)-1); return x[lo]+(x[hi]-x[lo])*(i-lo)
def summ(sel,key):
    v=[r[key] for r in sel]
    return (len(v),min(v),q(v,.25),st.median(v),q(v,.75),max(v),st.mean(v),sum(1 for a in v if a>.20)/len(v),sum(1 for a in v if a>.40)/len(v))
print('relative error = RMSE / DT  (DT is the table dwell time of the same atmosphere row)')
for wl,key in (('1.045 um (ELLA, solid-state-like)','rel1045'),('1.31525 um (COIL)','rel1315')):
    print('\n##',wl)
    print(f"{'method':24s} n   min   q25   med   q75   max  mean  >20%  >40%")
    for g,ps in G.items():
        sel=[r for r in rows if r['profile'] in ps]
        n,mn,a,m,b,mx,me,p20,p40=summ(sel,key)
        print(f"{g:24s}{n:3d} {mn:5.2f} {a:5.2f} {m:5.2f} {b:5.2f} {mx:5.2f} {me:5.2f} {p20:5.2f} {p40:5.2f}")
print('\n## by site, 1.045 um, GFS vs clim/ground-tab vs standard (median rel err)')
for site in ('WPAFB','Kuwait','Brunei'):
    out=[]
    for g in ('GFS 48/24/12 h','Climatology+ground tab','1976 US Standard'):
        sel=[r for r in rows if r['site']==site and r['profile'] in G[g]]
        out.append(f"{g}: med {st.median(r['rel1045'] for r in sel):.2f} max {max(r['rel1045'] for r in sel):.2f}")
    print(site,' | '.join(out))
print('\n## by path, 1.045 um, median rel err (all six)')
for path in ('top of boundary layer','3000 m'):
    sel=[r for r in rows if r['path']==path]
    print(path, f"{st.median(r['rel1045'] for r in sel):.2f}", 'GFS', f"{st.median(r['rel1045'] for r in sel if 'GFS' in r['profile']):.2f}")
print('\n## 1.045 um largest rel errors')
for r in sorted(rows,key=lambda r:-r['rel1045'])[:8]: print(r['table'],r['site'],r['path'],r['season'],r['profile'],r['dt1045_s'],r['rmse1045_s'],f"{r['rel1045']:.2f}")
print('\n## 1.045 um smallest rel errors')
for r in sorted(rows,key=lambda r:r['rel1045'])[:5]: print(r['table'],r['site'],r['path'],r['season'],r['profile'],r['dt1045_s'],r['rmse1045_s'],f"{r['rel1045']:.2f}")
print('\n## 1.31525 um largest rel errors')
for r in sorted(rows,key=lambda r:-r['rel1315'])[:6]: print(r['table'],r['site'],r['path'],r['season'],r['profile'],r['dt1315_s'],r['rmse1315_s'],f"{r['rel1315']:.2f}")
print('\nabsolute RMSE 1.045 um, GFS rows: min %.1f max %.1f; all rows: min %.1f max %.1f'%(min(r['rmse1045_s'] for r in rows if 'GFS' in r['profile']),max(r['rmse1045_s'] for r in rows if 'GFS' in r['profile']),min(r['rmse1045_s'] for r in rows),max(r['rmse1045_s'] for r in rows)))

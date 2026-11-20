"""Optimize sagittal and lateral balance feedback on training-only supine starts."""
from search_teacher import HERE,np,lib,xml,stand,pool,ptr,c_char_p,c_int
lib.evaluate_balance.argtypes=[c_char_p,c_int,ptr,ptr,ptr,ptr,ptr]
def run():
    base=np.load(HERE/"teachers_01/keys_1.npy")
    starts=pool[pool[:,0]==1,1:58]
    rng=np.random.default_rng(845032)
    mean=np.array([-.12,.4,.09, -.08,.55,.08, .12,.3,.5,.8,.1,.3])
    sigma0=np.array([.25,.7,.12,.17,.6,.11,.25,.4,.6,.75,.12,.5])
    sigma=sigma0.copy()
    out=HERE/"balance_01"
    out.mkdir(exist_ok=True)
    best=-np.inf
    def scores(gains,states):
        n=len(gains)
        result=np.zeros(n)
        lib.evaluate_balance(xml,n,np.ascontiguousarray(states),np.tile(base,(n,1,1)),stand,result,np.ascontiguousarray(gains))
        return result
    for gen in range(35):
        par=mean+sigma*rng.normal(size=(48,12))
        par[0]=mean
        group=starts[rng.choice(len(starts),3,replace=False)]
        s=scores(np.repeat(par,3,axis=0),np.tile(group,(len(par),1))).reshape(-1,3).mean(1)
        elite=par[np.argsort(s)[-8:]]
        win=par[s.argmax()]
        val=scores(np.tile(win,(24,1)),starts[::5][:24])
        if val.mean()>best:
            best=val.mean()
            np.save(out/"feedback.npy",win)
        print(gen,best,s.max(),np.percentile(val,[10,50,90]).tolist(),flush=True)
        mean=.2*mean+.8*elite.mean(0)
        sigma=np.maximum(.13*sigma0,.2*sigma+.8*elite.std(0))
if __name__=="__main__": run()


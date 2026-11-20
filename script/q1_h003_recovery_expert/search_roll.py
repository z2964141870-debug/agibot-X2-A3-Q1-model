"""Find asymmetric rolls to a repeatable face-up launch pose."""
from search_teacher import HERE,np,pool,keys,evaluate

def run():
    out=HERE/"rolls_01"
    out.mkdir(exist_ok=True)
    rng=np.random.default_rng(66578)
    base=np.tile([-.6,1.1,-.35,-.8,.7,.25,.7,0,0,0,0,0,0],(5,1))
    sigma=np.tile([.9,.7,.4,1.3,.7,.36,.7,.4,.85,.8,1.15,.7,.8],(5,1))
    models={k:(base.copy(),sigma.copy(),-np.inf) for k in (0,2,3)}
    for generation in range(50):
        for k,(mean,std,best) in list(models.items()):
            starts=pool[pool[:,0]==k]
            size=72
            p=mean+std*rng.normal(size=(size,5,13))
            p[0]=mean
            s=evaluate(p,np.repeat(starts[[generation]],size,axis=0),1)
            elite=p[np.argsort(s)[-10:]]
            winner=p[s.argmax()]
            valid=evaluate(np.repeat(winner[None],12,axis=0),starts[::10][:12],1)
            if valid.mean()>best:
                best=valid.mean()
                np.save(out/f"keys_{k}.npy",keys(winner))
            models[k]=(.15*mean+.85*elite.mean(0),(.2*std+.8*elite.std(0)).clip(.13*sigma,sigma),best)
            print(generation,k,best,flush=True)
if __name__=="__main__":run()


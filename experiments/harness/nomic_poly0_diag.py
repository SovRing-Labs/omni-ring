# DIAGNOSTIC (labelled, not a re-score of NOMIC-POLY-0): why did meaning-preserving phases lose capacity vs TRI-0?
import json, numpy as np
D, K = 5120, 16; rng = np.random.default_rng(0)
d = json.load(open('<your-data>/vault-indexer/index/default__vector_store.json'))['embedding_dict']
E = np.array(list(d.values()), dtype=np.float32); N, F = E.shape
W = np.exp(2j*np.pi*np.arange(K)/K).astype(np.complex64)
cos = (E/np.linalg.norm(E,axis=1,keepdims=True)); 
print("mean pairwise cos (sample):", float((cos[:500]@cos[:500].T)[np.triu_indices(500,1)].mean()))
def phases(X):
    R1 = rng.standard_normal((X.shape[1], D)).astype(np.float32); R2 = rng.standard_normal((X.shape[1], D)).astype(np.float32)
    return (np.round(np.arctan2(X@R2, X@R1)/(2*np.pi)*K) % K).astype(np.int16)
Xc = E - E.mean(0); U_, s, Vt = np.linalg.svd(Xc, full_matrices=False); Xw = (Xc @ Vt.T) / (s + 1e-6) * np.sqrt(N)
arms = {"whitened": phases(Xw), "random_id_atoms": rng.integers(0, K, (N, D)).astype(np.int16)}
pos_all = rng.integers(0, K, (400, D)).astype(np.int16)
out = {}
for name, P in arms.items():
    C = W[P]
    for M in (64, 128, 256):
        ex = tot = com = comok = 0
        for g in range(20):
            grp = rng.choice(N, M, replace=False); pos = pos_all[:M]
            z = W[(P[grp]+pos) % K].sum(0); s8 = 127/np.abs(np.r_[z.real, z.imag]).max()
            z = np.round(z.real*s8) + 1j*np.round(z.imag*s8)
            S = ((z[None]*np.conj(W[pos])) @ np.conj(C).T).real
            t2 = np.argsort(-S,1)[:,:2]
            for k in range(M):
                a,b = t2[k]; ok = np.array_equal(E[a],E[grp[k]]); ex+=ok; tot+=1
                if (S[k,a]-S[k,b])/S[k,a] >= 0.05: com+=1; comok+=ok
        out[f"{name}/{M}"] = dict(exact=ex/tot, coverage=com/tot, committed_precision=comok/com if com else None)
        print(name, M, out[f"{name}/{M}"], flush=True)
json.dump(out, open('experiments/results/nomic_poly0_diag.result.json','w'), indent=1)

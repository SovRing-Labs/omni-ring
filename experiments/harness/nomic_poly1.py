import json, time, numpy as np
D, K, SEED = 5120, 16, 1
rng = np.random.default_rng(SEED)
d = json.load(open('<your-data>/vault-indexer/index/default__vector_store.json'))['embedding_dict']
ids = list(d); E = np.array([d[i] for i in ids], dtype=np.float32); N, F = E.shape
R1 = rng.standard_normal((F, D)).astype(np.float32); R2 = rng.standard_normal((F, D)).astype(np.float32)
Xc = E - E.mean(0); _, s_, Vt = np.linalg.svd(Xc, full_matrices=False); Xw = ((Xc @ Vt.T) / (s_ + 1e-6) * np.sqrt(N)).astype(np.float32)
ang = np.arctan2(Xw @ R2, Xw @ R1)
P = (np.round(ang / (2*np.pi) * K) % K).astype(np.int16)          # N x D phases: the codebook on the torus
W = np.exp(2j*np.pi*np.arange(K)/K).astype(np.complex64)
C = W[P]                                                          # N x D unit complex
pos_all = rng.integers(0, K, (400, D)).astype(np.int16)
def q8(z):
    s = 127/np.abs(np.concatenate([z.real, z.imag])).max()
    return np.round(z.real*s).astype(np.int8), np.round(z.imag*s).astype(np.int8)
res = {"N": int(N), "F": int(F), "D": D}
for M in (64, 128, 256):
    ex = exd = com = comok = tot = 0; t_read = 0.0
    for g in range(40):
        grp = rng.choice(N, M, replace=False); pos = pos_all[:M]
        poly = (W[(P[grp] + pos) % K]).sum(0)
        re, im = q8(poly); poly = re.astype(np.float32) + 1j*im.astype(np.float32)
        t = time.perf_counter()
        U = poly[None, :] * np.conj(W[pos])                          # unbind every slot
        S = (U @ np.conj(C).T).real                                  # sift vs whole codebook: M x N
        t_read += time.perf_counter() - t
        top2 = np.argsort(-S, 1)[:, :2]
        for k in range(M):
            a, b = top2[k]; s1, s2 = S[k, a], S[k, b]
            ok = np.array_equal(E[a], E[grp[k]]); ex += ok; exd += (a == grp[k]); tot += 1
            if (s1 - s2) / s1 >= 0.05: com += 1; comok += ok
    res[M] = dict(exact=ex/tot, exact_id=exd/tot, coverage=com/tot, committed_precision=(comok/com if com else None),
                  ms_per_item=1000*t_read/tot, polygon_bytes=2*D, raw_bytes=M*F*4)
    print(M, res[M], flush=True)
json.dump(res, open('experiments/results/nomic_poly1.result.json', 'w'), indent=1, default=str)

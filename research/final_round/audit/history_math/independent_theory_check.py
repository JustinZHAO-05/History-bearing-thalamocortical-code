"""Independent reconstruction of the historical mathematical claims.

No historical analysis code is imported. Inputs are displayed integer matrices
and, for finite illustrations only, the archived full-precision matrices.
Outputs are written beside this script; historical directories are read-only.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
R2 = HERE.parents[2] / 'round2_consciousness'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def H(Q, t):
    n = Q.shape[0] // 2
    return expm(Q * t)[:n, n:].sum() / (2*n)

def buildQ(B):
    return np.block([[-np.diag(B.sum(1)), B],
                     [B.T, -np.diag(B.sum(0))]])

def exact_certificate():
    B = sp.Matrix([[2,2,3,3],[3,5,5,7],[4,6,8,12],[6,7,9,18]])
    a, d = B*sp.ones(4,1), B.T*sp.ones(4,1)
    Q = (-sp.diag(*a)).row_join(B).col_join(B.T.row_join(-sp.diag(*d)))
    Es = []
    for i in range(3):
        for j in range(3):
            u,v = sp.zeros(4,1), sp.zeros(4,1)
            u[i],u[3],v[j],v[3]=1,-1,1,-1
            V=u*v.T
            Es.append(sp.zeros(4).row_join(V).col_join(V.T.row_join(sp.zeros(4))))
    J=sp.Matrix([[k*sp.trace(Q**(k-1)*E) for E in Es] for k in range(2,8)])
    L=sp.Matrix([[2*(a.T*E[:4,4:]*d)[0] for E in Es]])
    z=sp.Matrix([-34757708423748,78972436728859,60442160225802,
                 -4537143581364,-44523815735181,-76816258993573,
                 5839071640743,0,0])
    cert=dict(J_rank=J.rank(), augmented_rank=J.col_join(L).rank(),
              first_six_columns_det=str(J[:,:6].det()),
              first_seven_augmented_columns_det=str(J.col_join(L)[:,:7].det()),
              Jz=[str(x) for x in J*z], Lz=str((L*z)[0]),
              gradient=list(map(int,list(L))),
              characteristic_coefficients=[str(x) for x in Q.charpoly().all_coeffs()],
              generator_row_sums=[str(x) for x in Q*sp.ones(8,1)])
    assert cert['J_rank']==6 and cert['augmented_rank']==7
    assert all(x=='0' for x in cert['Jz']) and int(cert['Lz'])!=0
    assert cert['first_six_columns_det']=='150676075875044966400000'
    assert cert['first_seven_augmented_columns_det']=='353574528412802457600000000'
    return cert

def finite_audit(name):
    p=R2/'results'/name/'results.json'
    raw=json.loads(p.read_text())
    entries=raw['ends'] if 'ends' in raw else [raw['minus'],raw['plus']]
    # Archived keys differ between the six- and eight-state generations.
    qs=[np.array(e['Q_per_ms'],dtype=float) for e in entries]
    Qm,Qp=qs;N=len(Qm);n=N//2
    gap=float(raw['gap_at_maximum_ms'])
    em=np.sort_complex(np.linalg.eigvals(Qm))
    ep=np.sort_complex(np.linalg.eigvals(Qp))
    hm,hp=H(Qm,gap),H(Qp,gap)
    mp.mp.dps=70
    mq=[mp.matrix([[mp.mpf(str(v)) for v in row] for row in Q]) for Q in qs]
    def mh(Q):
        E=mp.expm(Q*mp.mpf(str(gap)))
        return sum(E[i,j] for i in range(n) for j in range(n,N))/N
    dh=mh(mq[1])-mh(mq[0])
    cubic=(np.linalg.matrix_power(Qp,3)[:n,n:].sum()-
           np.linalg.matrix_power(Qm,3)[:n,n:].sum())/(6*N)
    report=dict(input_sha256=sha(p),dimension=N,gap_ms=gap,
                H_minus=float(hm),H_plus=float(hp),difference=float(hp-hm),
                relative_difference=float((hp-hm)/hm),
                difference_70digits=str(dh),
                double_versus_70digit_difference_error=float(abs(mp.mpf(str(hp-hm))-dh)),
                eigenvalue_max_difference=float(np.max(abs(ep-em))),
                cubic_coefficient=float(cubic),
                cubic_report_error=float(abs(cubic-raw['cubic_coefficient_per_ms3'])),
                displayed_recovery_gap_error=float(abs(hp-hm-raw['maximum_paired_difference'])),
                full_within_block_generator_difference=float(max(
                    np.max(abs(Qp[:n,:n]-Qm[:n,:n])),
                    np.max(abs(Qp[n:,n:]-Qm[n:,n:])))),
                row_sum_error=float(max(np.max(abs(Q.sum(1))) for Q in qs)),
                column_sum_error=float(max(np.max(abs(Q.sum(0))) for Q in qs)))
    if np.max(abs(Qm-Qm.T)) < 1e-14:
        eig, U=np.linalg.eigh(Qm)
        c=np.r_[np.ones(n),np.zeros(n)]
        nonzero=abs(eig)>1e-12
        q=(U[:,nonzero].T@c)**2/N
        spectral_H=.25-np.dot(q,np.exp(eig[nonzero]*gap))
        report.update(symmetric=True,spectral_response_error=float(abs(spectral_H-hm)),
                      spectral_weight_total=float(q.sum()),
                      first_weighted_moment=float(np.dot(-eig[nonzero],q)),
                      second_weighted_moment=float(np.dot(eig[nonzero]**2,q)))
    assert report['displayed_recovery_gap_error']<1e-12
    assert report['full_within_block_generator_difference']==0
    return report

def matched_clamp_not_isospectral():
    Qs=[]
    for u in [.00005,.00495]:
        B=np.array([[u,.005-u],[.005-u,.09+u]])
        Qs.append(buildQ(B))
    t=151.1940085
    hs=[H(Q,t) for Q in Qs]
    evgap=np.max(abs(np.linalg.eigvalsh(Qs[1])-np.linalg.eigvalsh(Qs[0])))
    h3=[np.linalg.matrix_power(Q,3)[:2,2:].sum()/4 for Q in Qs]
    return dict(H=hs,difference=hs[1]-hs[0],relative_difference=(hs[1]-hs[0])/hs[0],
                gap_ms=t,H3=h3,cubic_coefficient=(h3[1]-h3[0])/6,
                eigenvalue_max_difference=float(evgap),
                interpretation='Both clamps and dominant mode fixed; remaining eigenvalues differ.')

def arbitrary_common_reference():
    """Direct affine simulations demonstrate no lawful-reference requirement."""
    rng=np.random.default_rng(202611114)
    errors=[]
    unlawful_reference_sizes=[]
    for _ in range(60):
        n=7;k=3;rho=rng.uniform(.1,1.)
        Ms=[rng.normal(size=(n,n))*.1-np.eye(n) for _ in range(3)]
        Us=[expm(M*rng.uniform(.1,.6)) for M in Ms]
        bs=[rng.normal(size=n) for _ in range(3)]
        refs=[100*rng.normal(size=n) for _ in range(2)]
        xA,xB=rng.normal(size=(2,n));w=rng.normal(size=n)
        masks=[np.r_[np.ones(k),np.zeros(n-k)],np.r_[np.zeros(k),np.ones(n-k)]]
        def f(i,j):
            pair=np.stack([xA,xB])
            for stage in range(3):
                pair=pair@Us[stage].T+bs[stage]
                which=i if stage==0 else (j if stage==1 else None)
                if which is not None:
                    mask=masks[which]
                    pair=pair*(1-rho*mask)+rho*mask*refs[stage]
            return float((pair[0]-pair[1])@w)
        pair=np.array([[f(i,j) for j in range(2)] for i in range(2)])
        kappa=2-rho
        errors.append(abs(pair.sum()-kappa*kappa*f(None,None)))
        errors.append(abs(pair[0,1]-pair[1,0]-kappa*(f(0,None)-f(None,0))))
        unlawful_reference_sizes.append(max(np.linalg.norm(r) for r in refs))
    return dict(cases=60,maximum_identity_error=max(errors),
                arbitrary_reference_norm_range=[min(unlawful_reference_sizes),max(unlawful_reference_sizes)],
                conclusion='For shared affine dynamics, any pair-common reset value cancels from contrast. A lawful nonlinear tangent reference is a separate modeling requirement.')

def main():
    result=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                code_sha256=sha(Path(__file__)), historical_code_imported=False,
                exact=exact_certificate(),matched_clamp=matched_clamp_not_isospectral(),
                finite_pairs={name:finite_audit(name) for name in [
                    'full_spectrum_geometry_v1','full_spectrum_geometry_symmetric_v2',
                    'full_spectrum_geometry_symmetric_continuation_v3']},
                common_reference_correction=arbitrary_common_reference())
    (HERE/'independent_theory_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

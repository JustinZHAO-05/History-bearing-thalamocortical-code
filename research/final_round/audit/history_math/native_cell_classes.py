"""Read and validate released PPC classes without assuming contiguous cell types."""
from pathlib import Path
import sqlite3,pickle
import numpy as np

def load_native_ppc_classes(checkpoint):
    """Return sorted integer E/I target indices; no checkpoint is mutated.

    The source BLOB arrays are NumPy uint16. Their order is not uniformly sorted
    across fields. Sorting supplies a deterministic proposal ordering.
    """
    path=Path(checkpoint).resolve()
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        row=dict(db.execute('SELECT * FROM currentRNNParameters ORDER BY id DESC LIMIT 1').fetchone())
    finally:
        db.close()
    e=np.sort(np.asarray(pickle.loads(row['PPCexh_idx']),dtype=np.int64).reshape(-1))
    i=np.sort(np.asarray(pickle.loads(row['PPCinh_idx']),dtype=np.int64).reshape(-1))
    mask=np.asarray(pickle.loads(row['SynapseMask']))
    n=int(row['PPC_N'])
    assert len(e)==int(row['PPCexh_N']) and len(i)==int(row['PPCinh_N'])
    assert len(np.intersect1d(e,i))==0
    assert np.array_equal(np.sort(np.r_[e,i]),np.arange(n))
    assert np.all(mask[:,e]>=0) and np.all(mask[:,i]<=0)
    return e,i

"""Two-process TCP loopback replay of the graph solver's 4x1 message schedule.

This is a synthetic transport diagnostic at the observed iteration counts,
not the solver's complete per-round schedule or a distributed optimization run.
Only one factor-vector upload and one coefficient-vector download per block
are replayed. The historical CSV overcounted payload by a factor of two;
corrected reruns are written to a distinct filename and are not manuscript evidence.
"""
from __future__ import annotations
import multiprocessing as mp, socket, struct, time, csv
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; IN=RES/'repeated_real_splits.csv'
HOST='127.0.0.1'; PORT=0

def recvn(s,n):
    b=bytearray()
    while len(b)<n:
        x=s.recv(n-len(b))
        if not x: raise RuntimeError('socket closed during fixed-size frame')
        b.extend(x)
    return b

def server(port_q, sizes, p, rounds):
    srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); srv.bind((HOST,0)); srv.listen(1); port_q.put(srv.getsockname()[1])
    conn,_=srv.accept(); conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    coeff=np.zeros(p,dtype=np.float64).tobytes()
    for _ in range(rounds):
        for m in sizes: recvn(conn,m*8)
        for _m in sizes: conn.sendall(coeff)
    conn.close();srv.close()

def one(dataset, seed, weight, iterations, n, p, repeats=3):
    sizes=[len(x) for x in np.array_split(np.arange(n),4)]; total=[]
    payload=iterations*(sum(sizes)+4*p)*8
    for rep in range(repeats):
        q=mp.Queue(); proc=mp.Process(target=server,args=(q,sizes,p,iterations)); proc.start(); port=q.get(timeout=10)
        cli=socket.socket(socket.AF_INET,socket.SOCK_STREAM); cli.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        deadline=time.perf_counter()+10
        while True:
            try: cli.connect((HOST,port)); break
            except ConnectionRefusedError:
                if time.perf_counter()>deadline: raise
                time.sleep(.005)
        msgs=[np.zeros(m,dtype=np.float64).tobytes() for m in sizes]; t=time.perf_counter()
        for _ in range(iterations):
            for b in msgs: cli.sendall(b)
            for _m in sizes: recvn(cli,p*8)
        elapsed=time.perf_counter()-t; cli.close(); proc.join(timeout=30)
        if proc.exitcode!=0: raise RuntimeError(f'server failed: {proc.exitcode}')
        total.append(elapsed)
    return dict(dataset=dataset,split_seed=seed,M=4,J=1,weight=weight,iterations=iterations,application_payload_bytes=payload,median_loopback_seconds=float(np.median(total)),runs_seconds=';'.join(f'{x:.6f}' for x in total),round_trips=iterations,application_messages=iterations*8,scope='two-process TCP loopback replay; not end-to-end distributed solver timing')

def main():
    df=pd.read_csv(IN); rows=[]
    for ds in ('diabetes','breast_cancer'):
        sub=df[(df.dataset==ds)&(df.split_seed==2026)&(df.M==4)&(df.J==1)]
        n=int(sub.iloc[0].n_train); p=int(sub.iloc[0].p)
        for _,r in sub.iterrows(): rows.append(one(ds,2026,r.weight,int(r.iterations),n,p))
    out=RES/'tcp_transport_replay_corrected.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows: print(r)
if __name__=='__main__': main()

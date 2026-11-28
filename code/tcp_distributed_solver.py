"""Two-process TCP implementation of one full graph-splitting iteration.

The Worker process aggregates the four sample blocks, factor maps, edge proximal
solves and factor-side projection. The Coordinator owns the feature-anchor state
and performs the feature-side weighted projection. Only serialized NumPy vectors cross TCP.
Nominal latency and size-proportional serialization waits are added at the
coordinator application layer; this does not cap socket bandwidth and is not
OS-level tc/netem shaping or a physical multi-host benchmark.
"""
from __future__ import annotations
import argparse, csv, json, multiprocessing as mp, os, socket, struct, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from run_experiments import prepare_real_case
from hybrid_solver import GraphSplitting, make_regression
from stopping_checks import is_checkpoint

ROOT=Path(__file__).resolve().parents[1]
HOST='127.0.0.1'

def checked_gap(primal, dual):
    """Reject a non-finite certificate or a substantive weak-duality violation."""
    if not np.isfinite(primal) or not np.isfinite(dual):
        raise ArithmeticError('Non-finite primal or dual certificate')
    if dual-primal > 1e-9*max(1.,abs(primal)):
        raise ArithmeticError('Fenchel lower bound exceeds the feasible objective')
    return max(primal-dual,0.)

def require_single_feature_block(meta):
    if len(meta['p_sizes'])!=1 or any(j!=0 for j in meta['edge_to_feature']):
        raise NotImplementedError('The TCP implementation supports the 4x1 layout only')

def recv_exact(sock,n):
    chunks=[]; got=0
    while got<n:
        b=sock.recv(n-got)
        if not b: raise ConnectionError('peer closed socket')
        chunks.append(b); got+=len(b)
    return b''.join(chunks)

def send_array(sock,a):
    b=np.ascontiguousarray(a,dtype=np.float64).tobytes()
    sock.sendall(struct.pack('!Q',len(b))+b)
    return len(b)+8

def recv_array(sock):
    n=struct.unpack('!Q',recv_exact(sock,8))[0]
    b=recv_exact(sock,n)
    return np.frombuffer(b,dtype=np.float64).copy(),n+8

def worker_process(X,y,args,weights,rho,tol,maxiter,check_every,portq,resultq,start_event):
    problem=make_regression(X,y,row_parts=4,column_parts=1,**args)
    solver=GraphSplitting(problem,weights=weights,rho=rho)
    edge_ids=range(len(solver.edges))
    xes=[solver.xes[e] for e in edge_ids]
    xs=solver.xs
    meta={'port':None,'p_sizes':problem.block_sizes,'c':solver.c.tolist(),'d':solver.d.tolist(),
          'edge_sizes':[int(solver.xes[e].stop-solver.xes[e].start) for e in edge_ids],
          'edge_to_feature':[int(j) for a,j,A in solver.edges],
          'l1':problem.l1,'ridge':problem.ridge,'rho':rho,
          'n_x':sum(s.stop-s.start for s in xs),'n_xe':sum(s.stop-s.start for s in xes)}
    srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM);srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind((HOST,0));srv.listen(1)
    portq.put((srv.getsockname()[1],meta))
    conn,_=srv.accept();conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    # Coordinator owns the feature anchors; worker stores only edge copies.
    solver.eta[ np.concatenate([np.arange(s.start,s.stop) for s in xs]) ]=0
    solver.u[ np.concatenate([np.arange(s.start,s.stop) for s in xs]) ]=0
    # Start the measured loop only after the coordinator has initialized its state.
    start_event.wait()
    start=time.perf_counter();compute_s=0.;network_s=0.;messages=0;payload=0;history=[]
    for k in range(1,maxiter+1):
        tc=time.perf_counter()
        xi=solver.prox(solver.eta-solver.u)
        # Factor-side projection is local to the worker.
        eta_local=xi.copy()
        for a,sl in enumerate(solver.ys):
            es=solver.edge_by_factor[a]
            residual=xi[sl]-sum(xi[solver.yes[e]] for e in es)
            delta=residual/(1/solver.a[a]+sum(1/solver.b[e] for e in es))
            eta_local[sl]=xi[sl]-delta/solver.a[a]
            for e in es: eta_local[solver.yes[e]]=xi[solver.yes[e]]+delta/solver.b[e]
        compute_s += time.perf_counter()-tc
        edge_vec=np.concatenate([xi[s] for s in xes])
        tn=time.perf_counter();payload+=send_array(conn,edge_vec);messages+=1
        anchor_and_edges,nbytes=recv_array(conn);payload+=nbytes;messages+=1
        network_s += time.perf_counter()-tn
        n_anchor=meta['n_x'];eta_x=anchor_and_edges[:n_anchor]
        edge_eta=anchor_and_edges[n_anchor:]
        off=0
        for s in xes:
            m=s.stop-s.start;eta_local[s]=edge_eta[off:off+m];off+=m
        off=0
        for s in xs:
            m=s.stop-s.start
            # Not authoritative; the coordinator's copy is used for certificate below.
            off+=m
        solver.eta=eta_local
        # Update worker-owned dual coordinates. Feature anchors are absent here.
        worker_slices=list(solver.ys)+list(solver.yes)+list(xes)
        for sl in worker_slices: solver.u[sl]+=xi[sl]-solver.eta[sl]
        if is_checkpoint(k,check_every,maxiter):
            tc=time.perf_counter()
            blocks=[];off=0
            for p in meta['p_sizes']:
                blocks.append(eta_x[off:off+p].copy());off+=p
            multipliers=[-rho*solver.a[a]*solver.u[sl] for a,sl in enumerate(solver.ys)]
            primal=problem.objective(blocks);dual=problem.dual_bound(multipliers)
            gap=checked_gap(primal,dual);rel=gap/max(1.,abs(primal))
            compute_s+=time.perf_counter()-tc
            history.append((k,primal,dual,rel))
            if rel<=tol: break
    elapsed=time.perf_counter()-start
    coef=eta_x.copy()
    result={'weight':weights,'iterations':k,'objective':primal,'dual':dual,'gap':gap,'relative_gap':rel,
            'converged':bool(rel<=tol),'coef':coef,'worker_wall_s':elapsed,'worker_compute_s':compute_s,
            'worker_network_wait_s':network_s,'payload_bytes':payload,'messages':messages,
            'history':history,'worker_process_pid':os.getpid()}
    # Tell coordinator to close cleanly.
    conn.sendall(b'END');conn.close();srv.close();resultq.put(result)

def coordinator_process(port,meta,latency_ms,bandwidth_mbps,readyq):
    require_single_feature_block(meta)
    cli=socket.socket(socket.AF_INET,socket.SOCK_STREAM);cli.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    deadline=time.time()+20
    while True:
        try: cli.connect((HOST,port));break
        except ConnectionRefusedError:
            if time.time()>deadline: raise
            time.sleep(.005)
    eta_x=np.zeros(meta['n_x']);u_x=np.zeros_like(eta_x)
    # Feature anchor slices, with block-specific dimensions.
    p_sizes=meta['p_sizes'];c=np.asarray(meta['c']);d=np.asarray(meta['d']);rho=meta['rho']
    edge_sizes=meta['edge_sizes'];edge_to_feature=meta['edge_to_feature']
    delay=latency_ms/1000.;rate=bandwidth_mbps*1e6/8.
    coord_bytes=0;coord_messages=0;emulated_delay=0.;coord_compute=0.
    readyq.put({'pid':os.getpid()})
    while True:
        # END is a short sentinel; otherwise read a framed float64 vector.
        first=cli.recv(1)
        if not first: break
        if first==b'E':
            rest=recv_exact(cli,2)
            if rest==b'ND':break
            raise RuntimeError('bad end sentinel')
        # The frame header is 8 bytes; first byte already consumed.
        header=first+recv_exact(cli,7);n=struct.unpack('!Q',header)[0]
        raw=recv_exact(cli,n);edge_xi=np.frombuffer(raw,dtype=np.float64).copy()
        coord_bytes+=8+n;coord_messages+=1
        # Added nominal delay and size-proportional serialization waiting.
        t=time.perf_counter();time.sleep(delay+(n+8)/rate);emulated_delay+=delay+(n+8)/rate
        xi_x=np.empty_like(eta_x);off=0
        for j,p in enumerate(p_sizes):
            v=eta_x[off:off+p]-u_x[off:off+p]
            coeff=rho*c[j]
            xi_x[off:off+p]=np.sign(v)*np.maximum(np.abs(v)-meta['l1']/coeff,0)/(1+meta['ridge']/coeff)
            off+=p
        eta_x_new=xi_x.copy();edge_eta=[];eoff=0;poff=0
        for j,p in enumerate(p_sizes):
            incident=[e for e,q in enumerate(edge_to_feature) if q==j]
            total=c[j]*xi_x[poff:poff+p].copy()
            for e in incident:
                m=edge_sizes[e];total+=d[e]*edge_xi[eoff:eoff+m];eoff+=m
            denom=c[j]+sum(d[e] for e in incident)
            mean=total/denom
            eta_x_new[poff:poff+p]=mean
            poff+=p
        # Reconstruct projected edge copies using the same feature-block means.
        eoff=0;poff=0
        for j,p in enumerate(p_sizes):
            mean=eta_x_new[poff:poff+p]
            for e,q in enumerate(edge_to_feature):
                if q==j:
                    m=edge_sizes[e];edge_eta.append(mean);eoff+=m
            poff+=p
        edge_eta_vec=np.concatenate(edge_eta) if edge_eta else np.empty(0)
        u_x+=xi_x-eta_x_new;eta_x=eta_x_new
        reply=np.concatenate([eta_x,edge_eta_vec]).astype(np.float64,copy=False).tobytes()
        time.sleep(delay+(len(reply)+8)/rate);emulated_delay+=delay+(len(reply)+8)/rate
        cli.sendall(struct.pack('!Q',len(reply))+reply)
        coord_bytes+=8+len(reply);coord_messages+=1
        coord_compute+=time.perf_counter()-t
    cli.close()
    readyq.put({'coordinator_bytes':coord_bytes,'coordinator_messages':coord_messages,'emulated_network_delay_s':emulated_delay,'coordinator_service_elapsed_s':coord_compute})

def get_data(ds,seed):
    if ds=='diabetes':
        z=prepare_real_case(ds,split_seed=seed,save_split=False)
        return z['X'],z['y'],z['Xtest'],z['ytest'],z['args'],z.get('target_scale',1.)
    if ds!='breast_cancer':raise ValueError(f'Unsupported dataset: {ds}')
    raw=load_breast_cancer();idx=np.arange(len(raw.target));y=2*raw.target-1
    tr,te=train_test_split(idx,test_size=.3,stratify=y,random_state=seed)
    sc=StandardScaler().fit(raw.data[tr]);return sc.transform(raw.data[tr]),y[tr],sc.transform(raw.data[te]),y[te],dict(loss='logistic',loss_parameter=1,ridge=.01,fusion=0,l1=.001),1.

def one_run(ds,seed,weight,latency_ms,bw_mbps,tol=1e-7,maxiter=30000):
    if weight not in {'unit','budget'}:raise ValueError('Unknown weight rule')
    if not np.isfinite(latency_ms) or latency_ms<0 or not np.isfinite(bw_mbps) or bw_mbps<=0:
        raise ValueError('Delay must be nonnegative and serialization rate positive')
    if not np.isfinite(tol) or tol<=0 or maxiter<1:raise ValueError('Invalid stopping parameters')
    X,y,Xt,yt,args,ys=get_data(ds,seed)
    # Spawn prevents the coordinator from inheriting the parent's data arrays.
    ctx=mp.get_context('spawn')
    pq=ctx.Queue();rq=ctx.Queue();ready=ctx.Queue();start_event=ctx.Event()
    wp=ctx.Process(target=worker_process,args=(X,y,args,weight,1.,tol,maxiter,10,pq,rq,start_event));wp.start()
    port,meta=pq.get(timeout=30)
    cp=ctx.Process(target=coordinator_process,args=(port,meta,latency_ms,bw_mbps,ready));cp.start();coord_info=ready.get(timeout=30)
    start_event.set()
    try: wr=rq.get(timeout=600)
    except Exception as exc:
        for proc in (wp,cp):
            if proc.is_alive():proc.terminate()
            proc.join(timeout=2)
        raise RuntimeError(f'worker result unavailable; worker_exit={wp.exitcode}, coordinator_exit={cp.exitcode}') from exc
    wp.join(timeout=30);cp.join(timeout=30);coord_stats=ready.get(timeout=30)
    if wp.exitcode!=0 or cp.exitcode!=0:raise RuntimeError(f'process exit worker={wp.exitcode}, coordinator={cp.exitcode}')
    for q in (pq,rq,ready):
        q.close()
        q.join_thread()
    # Numerical equivalence check against the in-process full-vector implementation.
    ref=GraphSplitting(make_regression(X,y,row_parts=4,column_parts=1,**args),weights=weight,rho=1.)
    refout=ref.solve(tol=tol,maxiter=maxiter)
    coef_distance=float(np.linalg.norm(wr['coef']-refout['coef']))
    objective_difference=float(wr['objective']-refout['objective'])
    if not wr['converged'] or not refout['converged']:
        raise RuntimeError('TCP or reference solve did not reach the common gap target')
    if wr['iterations']!=refout['iterations'] or coef_distance>1e-8 or abs(objective_difference)>1e-9:
        raise ArithmeticError('TCP and in-process implementations disagree')
    expected_bytes=wr['iterations']*(72*X.shape[1]+16)
    if wr['payload_bytes']!=expected_bytes or wr['payload_bytes']!=coord_stats['coordinator_bytes']:
        raise ArithmeticError('TCP application byte ledger mismatch')
    pred=Xt@wr['coef']
    if args['loss']=='logistic':
        pp=expit(pred);yy=(yt+1)/2;metrics={'test_logloss':float(log_loss(yy,pp)),'test_auc':float(roc_auc_score(yy,pp)),'test_rmse':''}
    else:metrics={'test_rmse':float(np.sqrt(np.mean((pred-yt)**2))*ys),'test_logloss':'','test_auc':''}
    return dict(dataset=ds,seed=seed,weight=weight,latency_ms=latency_ms,bandwidth_mbps=bw_mbps,
        iterations=wr['iterations'],relative_gap=wr['relative_gap'],objective=wr['objective'],
        reference_iterations=refout['iterations'],reference_relative_gap=refout['relative_gap'],
        coefficient_distance_to_reference=coef_distance,objective_difference_to_reference=objective_difference,
        worker_wall_s=wr['worker_wall_s'],worker_compute_s=wr['worker_compute_s'],
        worker_network_wait_s=wr['worker_network_wait_s'],payload_bytes=wr['payload_bytes'],
        messages=wr['messages'],emulated_network_delay_s=coord_stats['emulated_network_delay_s'],
        coordinator_bytes=coord_stats['coordinator_bytes'],coordinator_messages=coord_stats['coordinator_messages'],
        coordinator_service_elapsed_s=coord_stats['coordinator_service_elapsed_s'],
        test_rmse=metrics['test_rmse'],test_logloss=metrics['test_logloss'],test_auc=metrics['test_auc'],
        converged=wr['converged'],worker_pid=wr['worker_process_pid'],coordinator_pid=coord_info['pid'],
        setup='two independent spawned OS processes; Worker aggregates four sample blocks; Coordinator owns feature-anchor state; TCP loopback; added nominal latency plus serialization-delay model; timer starts after coordinator readiness')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--datasets',nargs='+',choices=['diabetes','breast_cancer'],default=['diabetes'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[2026])
    ap.add_argument('--profiles',nargs='+',choices=['lan','wan'],default=['lan','wan'])
    ap.add_argument('--output',default=str(ROOT/'results'/'tcp_distributed_solver.csv'))
    a=ap.parse_args();profiles={'lan':(0.1,1000.),'wan':(5.,20.)};rows=[]
    for ds in a.datasets:
      for seed in a.seeds:
       for profile in a.profiles:
        lat,bw=profiles[profile]
        for weight in ('unit','budget'):
         r=one_run(ds,seed,weight,lat,bw,tol=1e-7,maxiter=10000)
         r['seed']=seed;r['profile']=profile;rows.append(r)
         print(ds,seed,profile,weight,'iter',r['iterations'],'gap',r['relative_gap'],'wall',round(r['worker_wall_s'],3),'bytes',r['payload_bytes'],flush=True)
         Path(a.output).parent.mkdir(parents=True,exist_ok=True)
         with open(a.output,'w',newline='') as f:
          w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('WROTE',a.output,'rows',len(rows))
if __name__=='__main__':main()

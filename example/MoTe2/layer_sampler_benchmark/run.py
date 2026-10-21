"""Portable controller: four logical replicas, one process per selected GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpus',default=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--optimizer',choices=['sr','kfac'],default='sr')
    parser.add_argument('--kinetic-mode',choices=['scan','forward_laplacian'],default='scan')
    parser.add_argument('--kfac-inverse-period',type=int,default=10)
    parser.add_argument('--reset-optimizer',action='store_true')
    parser.add_argument('--stage',choices=['all','init','validate','train','evaluate','pilot','compare','analyze'],default='all')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    gpus=[g.strip() for g in args.gpus.split(',') if g.strip()]
    if not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus):parser.error('Choose one to four distinct GPU IDs or UUIDs.')
    if args.threads<1:parser.error('--threads must be positive.')
    logs=out/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda',
               OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    (out/'launcher.json').write_text(json.dumps(dict(command=sys.argv,python=sys.executable,
        gpus=gpus,processes_per_gpu=1,threads_per_process=args.threads,source=str(HERE),
        optimizer=args.optimizer,kinetic_mode=args.kinetic_mode,kfac_inverse_period=args.kfac_inverse_period),indent=2))

    def stage(name,script,arguments=(),parallel=False):
        (out/'CURRENT_STAGE.txt').write_text(name+'\n')
        ranks=list(range(4)) if parallel else [0]
        for start in range(0,len(ranks),len(gpus)):
            active=[]
            try:
                for slot,rank in enumerate(ranks[start:start+len(gpus)]):
                    child_env=env.copy();child_env.update(CUDA_VISIBLE_DEVICES=gpus[slot],NQS_RANK=str(rank))
                    path=logs/f'{name}-rank{rank}-{time.time_ns()}.log'
                    stream=path.open('w')
                    command=[sys.executable,'-u',str(HERE/script),*map(str,arguments)]
                    print(name,'rank',rank,'GPU',gpus[slot],'log',path,flush=True)
                    process=subprocess.Popen(command,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
                    active.append((process,stream,path))
                codes=[]
                for process,stream,path in active:
                    code=process.wait();stream.close();codes.append(code)
                    if code:print('Stage failed:',path,'exit',code,flush=True)
                if any(codes):raise subprocess.CalledProcessError(next(c for c in codes if c),[name])
            except BaseException:
                for process,stream,_ in active:
                    if process.poll() is None:process.terminate()
                    process.wait();stream.close()
                raise

    def analyze():
        stage('analyze','analyze.py',['--out',out/'production'])
        stage('plot','plot_results.py',['--base',out])
        stage('report','report.py',['--base',out])
        stage('verify','verify_results.py',['--base',out])

    backend=['--kinetic-mode',args.kinetic_mode]
    chosen=args.stage
    if chosen in ['all','init']:
        if not (out/'bloch_shell5.npz').exists():stage('initialize','spectral.py',['--out',out/'bloch_shell5.npz'])
    if chosen in ['all','validate']:stage('validate_initial','validate.py',['--base',out,*backend])
    if chosen in ['all','train']:
        stage('train','train.py',['--base',out,'--refine','--minimum-steps',8000,'--max-steps',16000,
              '--plateau-threshold',.02,'--holdout-records',2048,'--optimizer',args.optimizer,
              '--kfac-inverse-period',args.kfac_inverse_period,*backend,
              *(['--reset-optimizer'] if args.reset_optimizer else [])],True)
    if chosen in ['all','evaluate']:
        stage('evaluate_candidates','select_checkpoint.py',['--base',out,'--evaluate',*backend],True)
        stage('freeze','select_checkpoint.py',['--base',out,*backend])
        stage('validate_trained','validate.py',['--base',out,'--checkpoint',out/'frozen/checkpoint.pkl',*backend])
    if chosen in ['all','pilot']:
        stage('pilot','benchmark_worker.py',['--base',out,'--pilot','--out',out/'pilot',*backend],True)
        stage('tune','tune_wolff.py',['--base',out])
    if chosen in ['all','compare']:
        stage('compare','benchmark_worker.py',['--base',out,*backend],True)
        try:analyze()
        except subprocess.CalledProcessError as error:
            if error.returncode!=3 or error.cmd!=['verify']:raise
            stage('extend','benchmark_worker.py',['--base',out,'--records',8192,'--reference-records',16384,*backend],True)
            analyze()
    elif chosen=='analyze':analyze()
    (out/'CURRENT_STAGE.txt').write_text('completed '+chosen+'\n')

if __name__=='__main__':main()

"""
Anad Trainer V2 - Optimized
Real PyTorch backpropagation + CPU multi-threading.

Usage:
    python train.py                    # fresh training
    python train.py --resume           # continue from checkpoint
    python train.py --steps 10000      # more steps
    python train.py --threads 16       # set CPU threads (default: all cores)
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.config import ANAD_NANO, ANAD_SMALL
from tokenizer.tokenizer import AnadTokenizer
from training.data_collector import AnadDataCollector
from training.trainer import PauseController, cosine_lr_schedule

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",   type=int,   default=5000)
    p.add_argument("--model",   type=str,   default="nano")
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--seqlen",  type=int,   default=128)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--threads", type=int,   default=0,   help="CPU threads (0=auto)")
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--datadir", type=str,   default="./training/data")
    p.add_argument("--outdir",  type=str,   default="./checkpoints")
    return p.parse_args()

def build_model(cfg, device):
    import torch, torch.nn as nn, torch.nn.functional as F, math
    class N(nn.Module):
        def __init__(self,d,e=1e-6):
            super().__init__(); self.w=nn.Parameter(torch.ones(d)); self.e=e
        def forward(self,x):
            return x/x.pow(2).mean(-1,keepdim=True).add(self.e).sqrt()*self.w
    class A(nn.Module):
        def __init__(self,c):
            super().__init__()
            self.nh,self.nkv,self.hd,self.g=c.n_heads,c.n_kv_heads,c.head_dim,c.n_heads//c.n_kv_heads
            self.wq=nn.Linear(c.dim,c.n_heads*c.head_dim,bias=False)
            self.wk=nn.Linear(c.dim,c.n_kv_heads*c.head_dim,bias=False)
            self.wv=nn.Linear(c.dim,c.n_kv_heads*c.head_dim,bias=False)
            self.wo=nn.Linear(c.n_heads*c.head_dim,c.dim,bias=False)
        def forward(self,x,mask):
            B,T,_=x.shape
            q=self.wq(x).view(B,T,self.nh,self.hd).transpose(1,2)
            k=self.wk(x).view(B,T,self.nkv,self.hd).transpose(1,2).repeat_interleave(self.g,1)
            v=self.wv(x).view(B,T,self.nkv,self.hd).transpose(1,2).repeat_interleave(self.g,1)
            a=(q@k.transpose(-2,-1))/math.sqrt(self.hd)+mask[:T,:T]
            return self.wo((F.softmax(a,-1)@v).transpose(1,2).reshape(B,T,-1))
    class FF(nn.Module):
        def __init__(self,c):
            super().__init__()
            self.w1=nn.Linear(c.dim,c.hidden_dim,bias=False)
            self.w2=nn.Linear(c.hidden_dim,c.dim,bias=False)
            self.w3=nn.Linear(c.dim,c.hidden_dim,bias=False)
        def forward(self,x):
            return self.w2(F.silu(self.w1(x))*self.w3(x))
    class Block(nn.Module):
        def __init__(self,c):
            super().__init__()
            self.a,self.f=A(c),FF(c); self.n1,self.n2=N(c.dim),N(c.dim)
        def forward(self,x,mask):
            x=x+self.a(self.n1(x),mask); return x+self.f(self.n2(x))
    class M(nn.Module):
        def __init__(self,c):
            super().__init__()
            self.emb=nn.Embedding(c.vocab_size,c.dim)
            self.layers=nn.ModuleList([Block(c) for _ in range(c.n_layers)])
            self.norm=N(c.dim); self.head=nn.Linear(c.dim,c.vocab_size,bias=False)
            self.head.weight=self.emb.weight
            m=torch.full((c.max_seq_len,c.max_seq_len),float('-inf'))
            self.register_buffer('mask',torch.triu(m,1))
        def forward(self,x):
            h=self.emb(x)
            for l in self.layers: h=l(h,self.mask)
            return self.head(self.norm(h))
    model=M(cfg).to(device)
    # Compile model for faster CPU execution (PyTorch 2.0+)
    try:
        model=torch.compile(model)
        print("  Model compiled (faster inference)")
    except Exception:
        pass
    n=sum(p.numel() for p in model.parameters())
    print(f"  Model: {cfg.model_name} {n/1e6:.1f}M params on {device.upper()}")
    return model

def get_batch(texts, tokenizer, batch, seqlen, device):
    import torch
    toks=[]
    for t in texts:
        try: toks.extend(tokenizer.encode(t))
        except: pass
    need=batch*seqlen+1
    while len(toks)<need: toks.extend(toks)
    xs,ys=[],[]
    for i in range(batch):
        s=(i*seqlen)%(len(toks)-seqlen-1)
        xs.append(toks[s:s+seqlen]); ys.append(toks[s+1:s+seqlen+1])
    return (torch.tensor(xs,dtype=torch.long).to(device),
            torch.tensor(ys,dtype=torch.long).to(device))

def save_ckpt(model, step, outdir):
    import torch
    # Unwrap compiled model if needed
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    d=os.path.join(outdir,f"checkpoint_step_{step:07d}")
    os.makedirs(d,exist_ok=True)
    torch.save({"step":step,"model":m.state_dict()},os.path.join(d,"model.pt"))
    print(f"  Checkpoint saved -> {d}")
    return d

def main():
    args=parse_args()
    cfg=ANAD_NANO if args.model=="nano" else ANAD_SMALL

    print("\n"+"="*52)
    print("  ANAD TRAINING V2 — Optimized")
    print("="*52)

    try:
        import torch, torch.nn.functional as F
        # ── CPU Optimization ──────────────────────────────────
        n_threads = args.threads if args.threads > 0 else os.cpu_count()
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(max(1, n_threads // 2))
        # Enable CPU performance optimizations
        torch.backends.mkldnn.enabled = True
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            gpu = torch.cuda.get_device_name(0)
            print(f"\n  GPU: {gpu}")
        else:
            print(f"\n  CPU: {n_threads} threads enabled")
            print(f"  PyTorch {torch.__version__} with MKL-DNN acceleration")
    except ImportError:
        print("\n  PyTorch not found. Run: pip install torch")
        return

    # Data
    col=AnadDataCollector(args.datadir)
    if args.collect or col.total_records()==0:
        print("\n  Collecting data...")
        col.collect_all(include_gutenberg=True,include_wikipedia=True,include_indic=True,max_records=10000)
        col._flush()

    # Add coding data if available
    try:
        from training.coding_data import CodingDataCollector
        cdc = CodingDataCollector(args.datadir)
        cdc.collect(include_stackoverflow=False)
    except Exception:
        pass

    texts=list(col.stream_for_training())
    print(f"  Training texts: {len(texts)}")
    if not texts:
        print("  No data. Run: python train.py --collect"); return

    # Tokenizer
    tp=os.path.join(args.outdir,"tokenizer")
    if os.path.exists(os.path.join(tp,"vocab.json")):
        print("  Loading tokenizer...")
        tokenizer=AnadTokenizer.load(tp)
    else:
        print("  Training tokenizer...")
        tokenizer=AnadTokenizer(vocab_size=8000)
        tokenizer.train(texts[:min(500,len(texts))])
        tokenizer.save(tp)
    print(f"  Vocab: {len(tokenizer.vocab)}")

    # Model + optimizer
    os.makedirs(args.outdir,exist_ok=True)
    model=build_model(cfg, dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.1,betas=(0.9,0.999))

    # Resume
    start=0
    if args.resume:
        ckpts=sorted([d for d in os.listdir(args.outdir) if d.startswith("checkpoint_step_")])
        if ckpts:
            pt=os.path.join(args.outdir,ckpts[-1],"model.pt")
            if os.path.exists(pt):
                ck=torch.load(pt,map_location=dev,weights_only=True)
                # Load into unwrapped model
                m=model._orig_mod if hasattr(model,'_orig_mod') else model
                m.load_state_dict(ck["model"]); start=ck["step"]
                print(f"  Resumed from step {start}")

    ctrl=PauseController(); ctrl.start()
    print(f"\n  Steps:{args.steps} Batch:{args.batch} SeqLen:{args.seqlen} Threads:{n_threads}")
    print("  P + Enter = pause\n")

    best,run,rc,t0,seen=999.0,0.0,0,time.time(),0
    model.train()

    for step in range(start,args.steps):
        ctrl.check()
        if ctrl.stop_requested: break

        i=step%max(1,len(texts))
        bt=texts[i:i+args.batch]
        if len(bt)<args.batch: bt=texts[:args.batch]
        x,y=get_batch(bt,tokenizer,args.batch,args.seqlen,dev)

        opt.zero_grad()
        loss=F.cross_entropy(model(x).view(-1,cfg.vocab_size),y.view(-1),ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (model._orig_mod if hasattr(model,'_orig_mod') else model).parameters(), 1.0
        )
        lr=cosine_lr_schedule(step,args.steps//10,args.steps,args.lr,args.lr*0.1)
        for pg in opt.param_groups: pg["lr"]=lr
        opt.step()

        lv=loss.item(); run+=lv; rc+=1; seen+=args.batch*args.seqlen
        if lv<best: best=lv

        if(step+1)%10==0:
            e=time.time()-t0
            sps=(step+1-start)/(e+1e-9)
            eta=(args.steps-step-1)/sps
            print(f"  step {step+1:5d}/{args.steps} | "
                  f"loss {run/rc:.4f} | "
                  f"lr {lr:.2e} | "
                  f"{sps:.1f} steps/s | "
                  f"ETA {int(eta//60)}m {int(eta%60)}s")
            run=rc=0

        if(step+1)%100==0:
            save_ckpt(model,step+1,args.outdir)

    save_ckpt(model,step+1,args.outdir); ctrl.stop()
    print(f"\n  Best loss: {best:.4f} | Tokens: {seen:,}")

    # Non-blocking signing
    try:
        import getpass
        ip=os.path.join("./anad_data","identity.json")
        if os.path.exists(ip):
            print("\n  Enter passphrase to sign (or press Enter to skip):")
            pw=getpass.getpass("  > ")
            if pw.strip():
                from node.identity import AnadIdentity
                idn=AnadIdentity.load(ip,pw)
                print(f"  Signed by {idn.node_id[:24]}...")
    except(KeyboardInterrupt,EOFError):
        pass

    print("\n  Done. Run: python main.py\n")

if __name__=="__main__":
    main()

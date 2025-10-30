
import os, json, glob, base64, argparse, asyncio, random, re, math, statistics
from io import BytesIO
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path

import numpy as np
import httpx
from PIL import Image, ImageDraw, ImageFont

# ---------- Config ----------
DEFAULT_PALETTE = {
    0:(0,0,0),1:(0,0,255),2:(255,0,0),3:(0,255,0),4:(255,255,0),
    5:(128,128,128),6:(255,0,255),7:(255,165,0),8:(0,128,128),9:(128,0,0)
}
CELL=24; GRID_PAD=6; ARROW_PAD=16; ARROW_WIDTH=28
OPENROUTER_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"

# ---------- Utilities ----------
def grid_to_text(grid:List[List[int]])->str:
    return "\\n".join(" ".join(str(v) for v in row) for row in grid)

def example_text_pair(inp:List[List[int]], out:List[List[int]])->str:
    return f"Input:\\n{grid_to_text(inp)}\\n\\nOutput:\\n{grid_to_text(out)}"

def question_text(inp:List[List[int]])->str:
    return f"Input:\\n{grid_to_text(inp)}\\n\\nOutput:\\n?"

def render_grid(grid:List[List[int]], palette:Dict[int,tuple]=DEFAULT_PALETTE)->Image.Image:
    H=len(grid); W=len(grid[0])
    img=Image.new("RGB",(W*CELL+2*GRID_PAD,H*CELL+2*GRID_PAD),(255,255,255))
    d=ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            x0=GRID_PAD + c*CELL; y0=GRID_PAD + r*CELL
            x1=x0+CELL-1; y1=y0+CELL-1
            color=palette.get(grid[r][c], (0,0,0))
            d.rectangle([x0,y0,x1,y1], fill=color, outline=(220,220,220))
    d.rectangle([0,0,img.width-1,img.height-1], outline=(200,200,200))
    return img

def render_question_box()->Image.Image:
    QUESTION_W=12*CELL; QUESTION_H=12*CELL
    img=Image.new("RGB",(QUESTION_W,QUESTION_H),(255,255,255))
    d=ImageDraw.Draw(img)
    d.rectangle([0,0,img.width-1,img.height-1],outline=(180,180,180))
    try: font=ImageFont.truetype("DejaVuSans.ttf",96)
    except: font=ImageFont.load_default()
    txt="?"
    bbox=d.textbbox((0,0),txt,font=font)
    tw=bbox[2]-bbox[0]; th=bbox[3]-bbox[1]
    d.text(((img.width-tw)//2,(img.height-th)//2),txt,fill=(0,0,0),font=font)
    return img

def compose_lr_arrow(left:Image.Image,right:Image.Image)->Image.Image:
    H=max(left.height,right.height)
    W=left.width+ARROW_PAD+ARROW_WIDTH+ARROW_PAD+right.width
    img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img)
    x=0
    img.paste(left,(x,(H-left.height)//2)); x+=left.width+ARROW_PAD
    y_mid=H//2; x0=x; x1=x+ARROW_WIDTH-10
    d.line((x0,y_mid,x1,y_mid),fill=(0,0,0),width=3)
    d.polygon([(x1,y_mid-7),(x1,y_mid+7),(x1+10,y_mid)],fill=(0,0,0))
    x+=ARROW_WIDTH+ARROW_PAD
    img.paste(right,(x,(H-right.height)//2))
    d.rectangle([0,0,img.width-1,img.height-1],outline=(200,200,200))
    return img

def pil_to_data_url(img:Image.Image)->str:
    buf=BytesIO(); img.save(buf,format="PNG")
    b64=base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def load_arc_tasks(dataset_dir:str, limit:Optional[int]=None)->List[Dict[str,Any]]:
    paths=sorted(glob.glob(os.path.join(dataset_dir,"*.json")))
    random.shuffle(paths)
    tasks=[]
    for p in paths[:limit] if limit else paths:
        with open(p,"r") as f:
            tasks.append(json.load(f))
    return tasks

# ---------- Validation helpers ----------
def is_grid(x)->bool:
    return isinstance(x,list) and len(x)>0 and all(isinstance(r,list) and len(r)>0 for r in x) and \
           len({len(r) for r in x})==1 and all(all(isinstance(v,int) and 0<=v<=9 for v in r) for r in x)

def grids_equal(a:List[List[int]], b:List[List[int]])->bool:
    return is_grid(a) and is_grid(b) and len(a)==len(b) and len(a[0])==len(b[0]) and all(a[r][c]==b[r][c] for r in range(len(a)) for c in range(len(a[0])))

def per_cell_accuracy(a:List[List[int]], b:List[List[int]])->float:
    if not (is_grid(a) and is_grid(b) and len(a)==len(b) and len(a[0])==len(b[0])): return 0.0
    H=len(a); W=len(a[0])
    match=sum(1 for r in range(H) for c in range(W) if a[r][c]==b[r][c])
    return match/(H*W)

# ---------- JSON robust extraction ----------
def extract_json_block(text:str)->str:
    text=text.strip()
    stack=[]
    start=None
    for i,ch in enumerate(text):
        if ch in '{[':
            if start is None:
                start=i
            stack.append(ch)
        elif ch in '}]' and stack:
            stack.pop()
            if not stack:
                end=i+1
                return text[start:end]
    return text

# ---------- Naive grid parser (baseline) ----------
def parse_multiple_grids(text:str, expected_k:int)->Tuple[List[List[List[int]]], List[List[List[int]]]]:
    block=extract_json_block(text)
    try:
        obj=json.loads(block)
        if isinstance(obj, list):
            first=obj; second=[]
        elif isinstance(obj, dict):
            first=obj.get("first_try", obj.get("answer", obj.get("outputs", [])))
            second=obj.get("second_try", [])
        else:
            first=[]; second=[]
    except Exception:
        first=[]; second=[]
    def coerce_to_k_grids(x):
        if expected_k==1 and is_grid(x):
            return [x]
        if isinstance(x,list):
            grids=[g for g in x if is_grid(g)]
            if len(grids)>=expected_k:
                return grids[:expected_k]
        return ["" for _ in range(expected_k)]
    first_k=coerce_to_k_grids(first)
    second_k=coerce_to_k_grids(second)
    if not second_k:
        second_k=["" for _ in range(expected_k)]
    if first_k[0] == "":
        print("First grid empty")
    if second_k[0] == "":
        print("Second grid empty")
    return first_k, second_k

# ---------- DSL + executor ----------
def exec_program(prog:Dict[str,Any])->Optional[List[List[int]]]:
    try:
        size=prog.get("grid_size")
        if not (isinstance(size,list) and len(size)==2 and all(isinstance(v,int) and v>0 for v in size)):
            return None
        H,W=size
        grid=[[0 for _ in range(W)] for _ in range(H)]
        ops=prog.get("ops", [])
        if not isinstance(ops, list): ops=[]
        for op in ops:
            if not isinstance(op, dict): continue
            name=op.get("op") or op.get("type")
            if name is None:
                if all(k in op for k in ("r","c","h","w","color")): name="fill_rect"
                elif all(k in op for k in ("r","c","color")): name="set"
            if name in ("fill_rect","fill"):
                r=op.get("r"); c=op.get("c"); h=op.get("h"); w=op.get("w"); color=op.get("color")
                if all(isinstance(v,int) for v in (r,c,h,w,color)) and h>0 and w>0 and 0<=color<=9:
                    r1=max(0,r); c1=max(0,c); r2=min(H, r+h); c2=min(W, c+w)
                    for rr in range(r1, r2):
                        row=grid[rr]
                        for cc in range(c1, c2):
                            row[cc]=color
            elif name in ("set","paint"):
                r=op.get("r"); c=op.get("c"); color=op.get("color")
                if all(isinstance(v,int) for v in (r,c,color)) and 0<=color<=9:
                    if 0<=r<H and 0<=c<W:
                        grid[r][c]=color
        return grid
    except Exception:
        print("Exec program failed")
        return None

def parse_and_exec_programs(raw_text:str, expected_k:int)->Tuple[List[Any], List[Any]]:
    block=extract_json_block(raw_text)
    def to_grids(x):
        if expected_k==1 and isinstance(x, dict):
            g=exec_program(x); return [g if is_grid(g) else ""]
        if isinstance(x, list):
            gs=[]
            for p in x[:expected_k]:
                g=exec_program(p) if isinstance(p, dict) else None
                gs.append(g if is_grid(g) else "")
            if len(gs)<expected_k: gs += [""]*(expected_k-len(gs))
            return gs[:expected_k]
        return [""]*expected_k
    try:
        obj=json.loads(block)
        if isinstance(obj, dict):
            first=obj.get("first_try", obj.get("answer", obj.get("programs", [])))
            second=obj.get("second_try", [])
        else:
            first=[]; second=[]
    except Exception:
        first=[]; second=[]
    return to_grids(first), to_grids(second)


def extract_op_counts_from_raw(raw_text:str, expected_k:int)->Tuple[List[int], List[int]]:
    """
    For DSL strategy: read the raw JSON and count 'ops' per program for first_try/second_try.
    Returns two lists of length K (fill with 0 if missing).
    """
    block=extract_json_block(raw_text)
    try:
        obj=json.loads(block)
    except Exception:
        return [0]*expected_k, [0]*expected_k

    def count_list(lst):
        counts=[]
        if isinstance(lst, list):
            for p in lst[:expected_k]:
                if isinstance(p, dict) and isinstance(p.get('ops',[]), list):
                    counts.append(len(p['ops']))
                else:
                    counts.append(0)
        return counts + [0]*max(0, expected_k-len(counts))

    if isinstance(obj, dict):
        first = obj.get('first_try', obj.get('answer', obj.get('programs', [])))
        second = obj.get('second_try', [])
        return count_list(first), count_list(second)
    return [0]*expected_k, [0]*expected_k

# ---------- Prompt builders ----------
def build_messages_for_task_grid(task:Dict[str,Any], mode:str)->Tuple[List[Dict], List[List[List[int]]]]:
    train_pairs=task["train"]; test_pairs=task["test"]
    k=len(test_pairs); assert k>=1
    examples_text=[]; example_images=[]
    for ex in train_pairs:
        inp=ex["input"]; out=ex["output"]
        examples_text.append(example_text_pair(inp,out))
        if mode=="multimodal":
            example_images.append(pil_to_data_url(compose_lr_arrow(render_grid(inp),render_grid(out))))
    q_text_lines=["Questions:"]; q_images=[]; golds=[]
    for i, ex in enumerate(test_pairs, start=1):
        inp=ex["input"]; golds.append(ex["output"])
        q_text_lines.append(f"Q{i}: {question_text(inp)}")
        if mode=="multimodal":
            q_images.append(pil_to_data_url(compose_lr_arrow(render_grid(inp),render_question_box())))
    system=(
        "You are solving ARC tasks. Each example shows an input grid and its correct output grid. "
        "Grids use digits 0-9 for colors. Infer the transformation and apply it to each question. "
        "Do not assume the output size; it can differ from the input. You get two tries per question."
    )
    user_parts=[{"type":"text","text":"Examples:"}]
    for i,txt in enumerate(examples_text, start=1):
        user_parts.append({"type":"text","text":f"(Example {i})\\n{txt}"})
        if mode=="multimodal":
            user_parts.append({"type":"image_url","image_url":{"url":example_images[i-1]}})
    user_parts.append({"type":"text","text":"\\n" + "\\n".join(q_text_lines)})
    if mode=="multimodal":
        for i,url in enumerate(q_images, start=1):
            user_parts.append({"type":"text","text":f"(Q{i} image)"})
            user_parts.append({"type":"image_url","image_url":{"url":url}})
    user_parts.append({"type":"text","text":
        "Return JSON only with keys 'first_try' and 'second_try'. Each is a list of K grids. "
        "A grid is a 2D array of digits 0..9. Example for K=2:\\n"
        "{\\n  \\\"first_try\\\": [ [[0,1],[1,0]], [[2]] ],\\n  \\\"second_try\\\": [ [[0,0],[0,0]], [[2]] ]\\n}\\n"
    })
    messages=[{"role":"system","content":system},{"role":"user","content":user_parts}]
    return messages, golds

def build_messages_for_task_dsl(task:Dict[str,Any], mode:str)->Tuple[List[Dict], List[List[List[int]]]]:
    train_pairs=task["train"]; test_pairs=task["test"]
    k=len(test_pairs); assert k>=1
    examples_text=[]; example_images=[]
    for ex in train_pairs:
        inp=ex["input"]; out=ex["output"]
        examples_text.append(example_text_pair(inp,out))
        if mode=="multimodal":
            example_images.append(pil_to_data_url(compose_lr_arrow(render_grid(inp),render_grid(out))))
    q_text_lines=["Questions (return programs):"]; q_images=[]; golds=[]
    for i, ex in enumerate(test_pairs, start=1):
        inp=ex["input"]; golds.append(ex["output"])
        q_text_lines.append(f"Q{i}: {question_text(inp)}")
        if mode=="multimodal":
            q_images.append(pil_to_data_url(compose_lr_arrow(render_grid(inp),render_question_box())))
    dsl_spec = (
        "PROGRAM FORMAT (JSON ONLY).\\n"
        "For each question, reply with a program that constructs the output grid using only rectangles and single-cell edits.\\n"
        "Use 0-based indexing.\\n"
        "{\\n"
        "  \\\"grid_size\\\": [H, W],              // REQUIRED; output size may differ from input\\n"
        "  \\\"ops\\\": [                          // applied in order; later ops overwrite earlier\\n"
        "    {\\\"op\\\":\\\"fill_rect\\\",\\\"r\\\":R0,\\\"c\\\":C0,\\\"h\\\":HGT,\\\"w\\\":WID,\\\"color\\\":K},\\n"
        "    {\\\"op\\\":\\\"set\\\",\\\"r\\\":R,\\\"c\\\":C,\\\"color\\\":K}\\n"
        "  ]\\n"
        "}\\n"
        "Shortcuts allowed: \\\"fill\\\" as alias of \\\"fill_rect\\\"; \\\"paint\\\" as alias of \\\"set\\\".\\n"
        "Colors K are digits 0..9.\\n"
        "Return JSON ONLY with keys 'first_try' and 'second_try', each a list of L programs.\\n"
        "Example for L=2:\\n"
        "{\\n"
        "  \\\"first_try\\\": [\\n"
        "    {\\\"grid_size\\\":[3,3],\\\"ops\\\":[{\\\"op\\\":\\\"fill_rect\\\",\\\"r\\\":0,\\\"c\\\":0,\\\"h\\\":3,\\\"w\\\":3,\\\"color\\\":0},{\\\"op\\\":\\\"set\\\",\\\"r\\\":1,\\\"c\\\":1,\\\"color\\\":1}]},\\n"
        "    {\\\"grid_size\\\":[1,1],\\\"ops\\\":[{\\\"op\\\":\\\"set\\\",\\\"r\\\":0,\\\"c\\\":0,\\\"color\\\":2}]}\\n"
        "  ],\\n"
        "  \\\"second_try\\\": [\\n"
        "    {\\\"grid_size\\\":[3,3],\\\"ops\\\":[{\\\"op\\\":\\\"fill_rect\\\",\\\"r\\\":0,\\\"c\\\":0,\\\"h\\\":3,\\\"w\\\":3,\\\"color\\\":0},{\\\"op\\\":\\\"set\\\",\\\"r\\\":1,\\\"c\\\":2,\\\"color\\\":1}]},\\n"
        "    {\\\"grid_size\\\":[1,1],\\\"ops\\\":[{\\\"op\\\":\\\"set\\\",\\\"r\\\":0,\\\"c\\\":0,\\\"color\\\":2}]}\\n"
        "  ]\\n"
        "}"
    )
    system=(
        "You are solving ARC tasks. Infer the transformation from examples. For each QUESTION, "
        "construct the output grid by specifying its size and a sequence of rectangle fills and single-cell edits. "
        "Do not assume the output size; choose it explicitly in the program. You get two tries per question."
    )
    user_parts=[{"type":"text","text":"Examples:"}]
    for i,txt in enumerate(examples_text, start=1):
        user_parts.append({"type":"text","text":f"(Example {i})\\n{txt}"})
        if mode=="multimodal":
            user_parts.append({"type":"image_url","image_url":{"url":example_images[i-1]}})
    user_parts.append({"type":"text","text":"\\n" + "\\n".join(q_text_lines)})
    if mode=="multimodal":
        for i,url in enumerate(q_images, start=1):
            user_parts.append({"type":"text","text":f"(Q{i} image)"})
            user_parts.append({"type":"image_url","image_url":{"url":url}})
    user_parts.append({"type":"text","text":"\\n" + dsl_spec})
    messages=[{"role":"system","content":system},{"role":"user","content":user_parts}]
    return messages, golds

# ---------- OpenRouter call ----------
def build_payload(model:str, messages:List[Dict], temperature:float)->Dict[str,Any]:
    return {"model":model,"messages":messages,"temperature":temperature}

async def call_openrouter_async(client:httpx.AsyncClient, payload:Dict[str,Any], api_key:str,
                                timeout_s:float, retries:int, backoff_base:float)->str:
    headers={
        "Authorization":f"Bearer {api_key}",
        "Content-Type":"application/json",
        "HTTP-Referer":"https://local-eval",
        "X-Title":"ARC-DSL Eval",
    }
    attempt=0
    while True:
        try:
            r=await client.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=timeout_s)
            data=r.json()
            if "choices" not in data: raise ValueError(data)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            attempt+=1
            if attempt>retries: raise
            await asyncio.sleep((backoff_base**attempt)+random.uniform(0,0.25))

# ---------- Evaluation ----------
async def eval_task_once(idx:int, total:int, sem:asyncio.Semaphore, client:httpx.AsyncClient,
                         model:str, messages:List[Dict], golds:List[List[List[int]]],
                         api_key:str, timeout_s:float, retries:int, backoff_base:float,
                         strategy:str):
    async with sem:
        k=len(golds)
        try:
            reply = await call_openrouter_async(client, build_payload(model, messages, 0.0),
                                                api_key, timeout_s, retries, backoff_base)
            ops_first=[]; ops_second=[]
            if strategy=="grid":
                first, second = parse_multiple_grids(reply, k)
            elif strategy=="dsl":
                first, second = parse_and_exec_programs(reply, k)
                if first[0] == "":
                    print("First program failed")
                if second[0] == "":
                    print("Second program failed")
                # also extract op counts
                ops_first, ops_second = extract_op_counts_from_raw(reply, k)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            per_pair_exact = []
            per_pair_pcell = []
            for i in range(k):
                g= golds[i]
                a= first[i]; b= second[i]
                ok = (is_grid(a) and grids_equal(a,g)) or (is_grid(b) and grids_equal(b,g))
                per_pair_exact.append(bool(ok))
                p1 = per_cell_accuracy(a,g) if is_grid(a) else 0.0
                p2 = per_cell_accuracy(b,g) if is_grid(b) else 0.0
                per_pair_pcell.append(max(p1,p2))
            score_exact = sum(per_pair_exact)/k
            score_pcell = sum(per_pair_pcell)/k
            return {
                "idx": idx,
                "strategy": strategy,
                "score_exact": score_exact,
                "score_pcell": score_pcell,
                "per_pair_exact": per_pair_exact,
                "per_pair_pcell": per_pair_pcell,
                "pred_first": first,
                "pred_second": second,
                "golds": golds,
                "raw": reply
            }
        except Exception as e:
            return {"idx": idx, "strategy": strategy, "error": str(e), "score_exact": 0.0, "score_pcell": 0.0, "golds": golds}

async def eval_tasks(args):
    api_key=os.environ.get("OPENROUTER_API_KEY")
    if not api_key: raise RuntimeError("Set OPENROUTER_API_KEY env var.")
    raw_tasks=load_arc_tasks(args.dataset_dir, args.limit)
    built_grid=[]; built_dsl=[]
    for t in raw_tasks:
        if args.strategy in ("grid","both"):
            m,g=build_messages_for_task_grid(t, args.mode); built_grid.append((m,g))
        if args.strategy in ("dsl","both"):
            m,g=build_messages_for_task_dsl(t, args.mode); built_dsl.append((m,g))
    sem=asyncio.Semaphore(args.concurrency)
    records=[]
    async with httpx.AsyncClient(http2=True) as client:
        tasks=[]
        total = len(raw_tasks)
        if args.strategy in ("grid","both"):
            for idx,(m,g) in enumerate(built_grid, start=1):
                tasks.append(eval_task_once(idx, total, sem, client, args.model, m, g,
                                            api_key, args.timeout, args.retries, args.backoff, "grid"))
        if args.strategy in ("dsl","both"):
            for idx,(m,g) in enumerate(built_dsl, start=1):
                tasks.append(eval_task_once(idx, total, sem, client, args.model, m, g,
                                            api_key, args.timeout, args.retries, args.backoff, "dsl"))
        for coro in asyncio.as_completed(tasks):
            rec=await coro
            records.append(rec)
            if "error" in rec:
                print(f"[{rec['idx']}/{total}] ({rec['strategy']}) ERROR: {rec['error']}")
            else:
                print(f"[{rec['idx']}/{total}] ({rec['strategy']}) exact={rec['score_exact']:.3f}  pcell={rec['score_pcell']:.3f}")
    out_path = Path(args.out or f"arc_interface_results_{args.strategy}.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r)+"\n")
    def summarize(strategy_name:str):
        subset=[r for r in records if r.get("strategy")==strategy_name and "error" not in r]
        if not subset: return None
        exact= [r["score_exact"] for r in subset]
        pcell= [r["score_pcell"] for r in subset]
        return {
            "n_tasks": len(subset),
            "mean_exact": float(statistics.mean(exact)) if exact else 0.0,
            "mean_pcell": float(statistics.mean(pcell)) if pcell else 0.0
        }
    sum_grid = summarize("grid")
    sum_dsl  = summarize("dsl")
    print("\\n====== SUMMARY ======")
    if sum_grid:
        print(f"GRID  - tasks={sum_grid['n_tasks']}  mean_exact={sum_grid['mean_exact']:.3f}  mean_pcell={sum_grid['mean_pcell']:.3f}")
    if sum_dsl:
        print(f"DSL   - tasks={sum_dsl['n_tasks']}   mean_exact={sum_dsl['mean_exact']:.3f}   mean_pcell={sum_dsl['mean_pcell']:.3f}")
    if sum_grid and sum_dsl:
        by_idx={ (r['idx'],r['strategy']): r for r in records if 'error' not in r}
        exact_diffs=[]; pcell_diffs=[]
        for i in range(1, len(raw_tasks)+1):
            g=by_idx.get((i,'grid')); d=by_idx.get((i,'dsl'))
            if g and d:
                exact_diffs.append(d['score_exact'] - g['score_exact'])
                pcell_diffs.append(d['score_pcell'] - g['score_pcell'])
        if exact_diffs:
            mean_diff_exact=float(statistics.mean(exact_diffs))
            mean_diff_pcell=float(statistics.mean(pcell_diffs))
            print(f"Paired mean improvement (DSL - GRID): exact={mean_diff_exact:+.3f}, pcell={mean_diff_pcell:+.3f}")
    # --- Area-based analysis when both strategies are present ---
    if sum_grid and sum_dsl:
        # Build per-pair table: (area, pcell_grid, pcell_dsl, exact_grid, exact_dsl)
        rows=[]
        # Map (idx,strategy) -> record
        by_idx={ (r['idx'], r['strategy']): r for r in records if 'error' not in r }
        for i in range(1, len(raw_tasks)+1):
            g=by_idx.get((i,'grid')); d=by_idx.get((i,'dsl'))
            if not (g and d): continue
            golds = g['golds']
            for j, gold in enumerate(golds):
                area = len(gold)*len(gold[0]) if (gold and gold[0]) else 0
                pcell_grid = g['per_pair_pcell'][j]
                pcell_dsl  = d['per_pair_pcell'][j]
                exact_grid = 1.0 if g['per_pair_exact'][j] else 0.0
                exact_dsl  = 1.0 if d['per_pair_exact'][j] else 0.0
                rows.append((area, pcell_grid, pcell_dsl, exact_grid, exact_dsl))

        def bin_area(a):
            if a<=50: return 'small<=50'
            if a<=150: return 'medium<=150'
            return 'large>150'

        bins={}
        for area, pg, pd, eg, ed in rows:
            b=bin_area(area)
            bins.setdefault(b, []).append((pg,pd,eg,ed))

        print("\\n-- Area-binned mean accuracies --")
        for b in ['small<=50','medium<=150','large>150']:
            vals=bins.get(b, [])
            if not vals: continue
            mean_pg=float(statistics.mean(v[0] for v in vals))
            mean_pd=float(statistics.mean(v[1] for v in vals))
            mean_eg=float(statistics.mean(v[2] for v in vals))
            mean_ed=float(statistics.mean(v[3] for v in vals))
            print(f"{b:>12}: per-cell GRID={mean_pg:.3f} DSL={mean_pd:.3f} | exact GRID={mean_eg:.3f} DSL={mean_ed:.3f}")
        # --- DSL compression stats (ops per program) using the better of two tries ---
        if sum_dsl:
            dsl_rows=[]  # (area, ops_best)
            for i in range(1, len(raw_tasks)+1):
                d=by_idx.get((i,'dsl'))
                if not d: continue
                golds = d['golds']
                for j, gold in enumerate(golds):
                    area = len(gold)*len(gold[0]) if (gold and gold[0]) else 0
                    # choose the better try by per-cell metric
                    p1 = d['per_pair_pcell'][j]
                    p2 = p1  # default
                    if 'pred_second' in d:
                        p2 = d['per_pair_pcell'][j]  # same array; best-of-two already computed above
                    # but we still need to decide index; compare per-cell of first vs second individually
                    # We don't have individual per-cell per try, so approximate: if second_try is exact, prefer second; else first.
                    # If both exact or both non-exact, we fall back to ops_first (conservative).
                    # Note: This keeps code simple; for deeper analysis, store per-try pcell next time.
                    ops = d['ops_first'][j] if d['ops_first'] else 0
                    # prefer second if its grid matches exactly
                    second_is_exact = 1.0 if d['per_pair_exact'][j] and not grids_equal(d['pred_first'][j], gold) and grids_equal(d['pred_second'][j], gold) else 0.0
                    if second_is_exact:
                        ops = d['ops_second'][j]
                    dsl_rows.append((area, ops))

            if dsl_rows:
                # simple linear fit ops ~= a + b * area to see scaling
                areas=[a for a,_ in dsl_rows]
                opss =[o for _,o in dsl_rows]
                mean_area = float(statistics.mean(areas))
                mean_ops  = float(statistics.mean(opss))
                cov = sum((a-mean_area)*(o-mean_ops) for a,o in dsl_rows) / max(1, (len(dsl_rows)-1))
                var = sum((a-mean_area)**2 for a in areas) / max(1, (len(areas)-1))
                slope = cov/var if var>0 else 0.0
                print(f"DSL ops vs area: mean_ops={mean_ops:.2f}, mean_area={mean_area:.2f}, slope≈{slope:.4f} ops per cell")


def main():
    parser=argparse.ArgumentParser(description="ARC eval: naive grid vs. rectangle/square DSL interface.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.6)
    parser.add_argument("--mode", choices=["multimodal","text"], default="multimodal")
    parser.add_argument("--strategy", choices=["grid","dsl","both"], default="both")
    parser.add_argument("--out", type=str, default=None, help="Path to write JSONL results")
    args=parser.parse_args()
    random.seed(42)
    asyncio.run(eval_tasks(args))
if __name__=="__main__":
    main()
